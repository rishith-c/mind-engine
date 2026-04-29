-- Mind-Engine schema. Storage only — NO pgvector. The from-scratch constraint
-- forbids any vector DB extension. Embeddings are stored as raw bipolar bytes
-- and similarity is computed in the application layer with our custom
-- GeometricIndex (packages/ai-cognitron/cognitron/geometric_index.py).
--
-- Two namespaces: cognitron_* and morpheus_*

-- =========================================================================
-- COGNITRON (Particle Neural Network) — persisted thought field
-- =========================================================================

create table if not exists cognitron_particles (
    id            bigserial primary key,
    text          text       not null,
    embedding     bytea      not null,             -- 10000-dim bipolar packed as int8
    embedding_dim int        not null default 10000,
    position_x    real       not null,
    position_y    real       not null,
    position_z    real       not null,
    mass          real       not null default 1.0,
    charge        real       not null default 0.5,
    polarity      smallint   not null default 1,
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now(),
    last_accessed timestamptz not null default now()
);

create index if not exists cognitron_particles_mass on cognitron_particles (mass desc);
create index if not exists cognitron_particles_recent on cognitron_particles (last_accessed desc);

create table if not exists cognitron_query_log (
    id           bigserial primary key,
    query_text   text       not null,
    hit_ids      bigint[]   not null,
    hit_scores   real[]     not null,
    created_at   timestamptz not null default now()
);

-- =========================================================================
-- MORPHEUS (3D synesthetic NCA) — persisted growth sessions
-- =========================================================================

create table if not exists morpheus_seeds (
    id          bigserial primary key,
    target      text       not null,                -- 'sphere', 'helix', ...
    grid_size   int        not null default 32,
    n_channels  int        not null default 16,
    weights_url text,                               -- pointer to weight blob (Storage)
    created_at  timestamptz not null default now()
);

create table if not exists morpheus_frames (
    id          bigserial primary key,
    seed_id     bigint     not null references morpheus_seeds(id) on delete cascade,
    t           int        not null,
    voxels      bytea      not null,                -- (D, H, W, 4) float16 packed
    audio_field bytea,                              -- (D, H, W) float16 frequencies
    created_at  timestamptz not null default now(),
    unique (seed_id, t)
);

create index if not exists morpheus_frames_seed_t on morpheus_frames (seed_id, t);

-- =========================================================================
-- Decay job — runs every hour, mass-decays old particles
-- =========================================================================

create or replace function cognitron_decay_step()
returns void language sql as $$
    update cognitron_particles
    set mass = mass * 0.985,
        updated_at = now()
    where last_accessed < now() - interval '1 hour';

    delete from cognitron_particles
    where mass < 0.001;
$$;

-- =========================================================================
-- Row-level security: open for demo, lock down for prod
-- =========================================================================

alter table cognitron_particles enable row level security;
alter table cognitron_query_log enable row level security;
alter table morpheus_seeds      enable row level security;
alter table morpheus_frames     enable row level security;

-- For the hackathon demo we allow public read/write. In production these
-- would be replaced with auth.uid()-scoped policies.
create policy "demo_open_read" on cognitron_particles for select using (true);
create policy "demo_open_write" on cognitron_particles for insert with check (true);
create policy "demo_open_update" on cognitron_particles for update using (true);
create policy "demo_open_log_write" on cognitron_query_log for insert with check (true);
create policy "demo_open_log_read"  on cognitron_query_log for select using (true);
create policy "demo_open_seed_write" on morpheus_seeds for insert with check (true);
create policy "demo_open_seed_read"  on morpheus_seeds for select using (true);
create policy "demo_open_frame_write" on morpheus_frames for insert with check (true);
create policy "demo_open_frame_read"  on morpheus_frames for select using (true);
