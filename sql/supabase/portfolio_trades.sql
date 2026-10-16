-- Per-user trade ledger for the Portfolio tab (run once in the Supabase SQL editor). Append-only: every buy, sell,
-- rebalance and manual edit with its price, fees and the cash left after it (core/ledger.py).
create table if not exists public.portfolio_trades (
    id          bigint generated always as identity primary key,
    user_id     uuid        not null,
    traded_at   timestamptz not null,
    ticker      text        not null,
    -- ADJUST = position set by hand (opening balance / correction): qty = shares after, price = unit cost basis after
    side        text        not null check (side in ('BUY', 'SELL', 'ADJUST')),
    qty         double precision not null check (qty >= 0),
    price       double precision not null check (price >= 0),
    fees        double precision not null default 0,
    cash_after  double precision,
    source      text        not null default 'manual',
    note        text,
    created_at  timestamptz not null default now()
);
create index if not exists portfolio_trades_user_idx on public.portfolio_trades (user_id, traded_at);

-- The dashboard uses the service-role key and filters by user_id itself; RLS still protects the
-- table from the anon key.
alter table public.portfolio_trades enable row level security;

-- Tables created before ADJUST existed (run once):
--   alter table public.portfolio_trades drop constraint if exists portfolio_trades_side_check;
--   alter table public.portfolio_trades add constraint portfolio_trades_side_check check (side in ('BUY', 'SELL', 'ADJUST'));
--   alter table public.portfolio_trades drop constraint if exists portfolio_trades_qty_check;
--   alter table public.portfolio_trades drop constraint if exists portfolio_trades_price_check;
