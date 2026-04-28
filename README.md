CREATE INDEX idx_pnl_ts
ON pnl_timeseries (ts DESC);

CREATE INDEX idx_pnl_portfolio_book_type_ts
ON pnl_timeseries (portfolio, book, pnl_type, ts DESC);

CREATE INDEX idx_pnl_desk_ts
ON pnl_timeseries (desk, ts DESC);

SELECT create_hypertable('pnl_timeseries', 'ts');


ALTER TABLE pnl_timeseries SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'desk, portfolio, book, pnl_type'
);



CREATE TABLE pnl_timeseries (
    ts              timestamptz NOT NULL,
    desk            text        NOT NULL,
    portfolio       text        NOT NULL,
    book            text        NOT NULL,
    underlying      text,
    product_type    text,
    currency        text        NOT NULL DEFAULT 'EUR',

    pnl_type        text        NOT NULL,
    pnl_value       numeric(18,4) NOT NULL,

    source_file     text,
    inserted_at     timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (
        ts,
        desk,
        portfolio,
        book,
        pnl_type
    )
);

Mon projet professionnel est de devenir trader en environnement global macro.

Je travaille actuellement en trading support, avec un rôle très orienté développement, où je conçois et maintiens des outils qui contribuent directement à améliorer la qualité d’exécution et la rentabilité du desk.

Mon travail consiste notamment à automatiser des processus front office et à exploiter les données pour analyser et optimiser la performance des stratégies de trading. Cela m’amène à être en interaction quotidienne avec les traders et à comprendre leurs contraintes, leurs prises de décision et les dynamiques de marché.

Cette position m’a permis de développer une vision concrète du fonctionnement d’un desk, à la fois sur les aspects techniques et financiers.

Mon objectif est désormais de me rapprocher de la prise de décision en marché, en capitalisant sur cette double compétence, pour évoluer vers un rôle de trader


# piptools-demo

Demo project to test pip-tools with multiple libraries.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install pip-tools
pip-compile requirements.in
pip-compile dev-requirements.in

pip install -r requirements.txt -r dev-requirements.txt
