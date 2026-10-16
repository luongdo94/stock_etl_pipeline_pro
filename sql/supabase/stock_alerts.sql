-- Per-user alert rules for the dashboard's Alert Center (run once in the Supabase SQL editor).
create table if not exists public.stock_alerts (
    id          bigint generated always as identity primary key,
    user_id     uuid        not null,
    ticker      text        not null,
    metric      text        not null check (metric in ('Price', 'Volume', 'Daily Return %', 'RSI')),
    condition   text        not null check (condition in ('above', 'below')),
    threshold   double precision not null,
    created_at  timestamptz not null default now()
);
create index if not exists stock_alerts_user_idx on public.stock_alerts (user_id);

-- The dashboard uses the service-role key and filters by user_id itself; RLS still protects the
-- table from the anon key.
alter table public.stock_alerts enable row level security;
