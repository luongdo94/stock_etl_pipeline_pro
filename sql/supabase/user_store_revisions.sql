-- Install after replace_user_rows.sql; safe to run repeatedly in the Supabase SQL editor.
create table if not exists public.user_store_revisions (
    user_id uuid not null,
    store text not null,
    rev bigint not null default 0,
    primary key (user_id, store)
);
alter table public.user_store_revisions enable row level security;

create or replace function public.get_user_store_rev(p_user_id uuid, p_store text)
returns bigint
language sql
security definer
set search_path = public
as $$
    select coalesce((select rev from public.user_store_revisions
                     where user_id = p_user_id and store = p_store), 0::bigint);
$$;

create or replace function public.replace_user_rows_rev(
    p_table text, p_user_id uuid, p_rows jsonb, p_expected_rev bigint)
returns bigint
language plpgsql
security definer
set search_path = public
as $$
declare
    current_rev bigint;
begin
    if p_table not in ('stock_watchlist', 'stock_portfolio') or p_table is null then
        raise exception 'table % not allowed', p_table;
    end if;
    insert into public.user_store_revisions (user_id, store)
    values (p_user_id, p_table) on conflict (user_id, store) do nothing;
    select rev into current_rev from public.user_store_revisions
    where user_id = p_user_id and store = p_table for update;
    if current_rev is distinct from p_expected_rev then
        raise exception 'stale revision: expected %, current %', p_expected_rev, current_rev;
    end if;
    perform public.replace_user_rows(p_table, p_user_id, p_rows);
    update public.user_store_revisions set rev = current_rev + 1
    where user_id = p_user_id and store = p_table;
    return current_rev + 1;
end;
$$;

-- Same service-role-only execution boundary as replace_user_rows.
revoke all on function public.replace_user_rows_rev(text, uuid, jsonb, bigint) from public, anon, authenticated;
revoke all on function public.get_user_store_rev(uuid, text) from public, anon, authenticated;
grant execute on function public.replace_user_rows_rev(text, uuid, jsonb, bigint) to service_role;
grant execute on function public.get_user_store_rev(uuid, text) to service_role;
