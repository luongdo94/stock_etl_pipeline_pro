-- Atomic replace of one user's watchlist / portfolio rows (audit A08 / R2-04). Run once in the Supabase SQL editor
-- and verify with the checks at the bottom. Called by services/user_store._replace_user_rows (no client fallback).
-- * Explicit column lists: id / created_at / other columns keep their DEFAULTs (no jsonb_populate_recordset NULLs).
-- * Every inserted row is forced to p_user_id; rows claiming another user raise an error.
-- * ADJUST the p_user_id type (uuid vs text) and the column types to your actual schema before installing.
create or replace function public.replace_user_rows(p_table text, p_user_id uuid, p_rows jsonb)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  if exists (select 1 from jsonb_array_elements(coalesce(p_rows, '[]'::jsonb)) r
             where r ? 'user_id' and (r->>'user_id') is distinct from p_user_id::text) then
    raise exception 'rows for another user';
  end if;
  if p_table = 'stock_watchlist' then
    delete from stock_watchlist where user_id = p_user_id;
    insert into stock_watchlist (user_id, ticker, status, thesis, catalyst, entry_target, invalidation_level,
                                 take_profit, next_earnings, added_date)
    select p_user_id, x.ticker, x.status, x.thesis, x.catalyst, x.entry_target, x.invalidation_level,
           x.take_profit, x.next_earnings,
           nullif(nullif(x.added_date, ''), 'None')::timestamptz   -- the column is timestamptz; the app sends text
    from jsonb_to_recordset(coalesce(p_rows, '[]'::jsonb)) as x(ticker text, status text, thesis text, catalyst text,
         entry_target double precision, invalidation_level double precision, take_profit double precision,
         next_earnings text, added_date text);
  elsif p_table = 'stock_portfolio' then
    delete from stock_portfolio where user_id = p_user_id;
    insert into stock_portfolio (user_id, ticker, shares, cost_basis)
    select p_user_id, x.ticker, x.shares, x.cost_basis
    from jsonb_to_recordset(coalesce(p_rows, '[]'::jsonb)) as x(ticker text, shares double precision,
         cost_basis double precision);
  else
    raise exception 'table % not allowed', p_table;
  end if;
end;
$$;
revoke all on function public.replace_user_rows(text, uuid, jsonb) from public, anon, authenticated;
-- Verification (staging project): call with a valid payload, then with a payload containing a bad ticker type and
-- confirm the original rows are still there; confirm anon/authenticated cannot execute the function.
-- Concurrency: last writer wins. Add a revision column + check if several sessions edit the same portfolio.
