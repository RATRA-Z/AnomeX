# AnomeX — Dashboard Overview (Frontend Only)

AI-Driven Anomaly Detection in Component Burn-In and Screening.
**This is UI-only**: no ML models, anomaly-detection logic, or backend
services are implemented. All numbers come from `mock/mock_data.py`.

## Run it

```bash
cd AnomeX
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```
AnomeX/
├── app.py                 # entry point, routing between pages
├── pages/
│   ├── overview.py         # the full Overview dashboard
│   └── coming_soon.py      # placeholder for all other nav items
├── ui/
│   ├── theme.py             # ALL colors + global CSS (contrast-checked)
│   ├── sidebar.py            # brand, nav, dataset panel, uploader
│   ├── cards.py               # header controls + KPI cards + CTA banner
│   ├── charts.py                # Plotly donut + health trend line
│   └── tables.py                  # high-risk & lot summary tables
├── mock/
│   └── mock_data.py         # single source of truth for every number
└── requirements.txt
```

## Why the numbers always agree

Every figure on the page (KPI cards, donut chart + percentages, lot
table roll-ups) is derived from one dict of health-category counts in
`mock/mock_data.py`. Percentages are calculated (`count / total * 100`),
never typed in, and the module asserts that:

- the four health categories sum to `TOTAL_COMPONENTS`
- per-lot components sum to `TOTAL_COMPONENTS`
- per-lot High Risk / Critical sums equal the top-level KPI counts

If you (or a future backend) ever change one number without updating
the others, these asserts will fail loudly instead of letting the
dashboard silently disagree with itself — which was the exact bug in
the original reference design (KPI said 182/30, the donut said
1,300/260).

## Swapping in real data later

Replace the contents of `mock/mock_data.py` with real Pandas
DataFrames / backend calls. Keep the same variable names and column
names (`Component ID`, `Lot ID`, `Health Score`, etc.) and the UI layer
needs no changes.

## Design notes

Dark theme built with an explicit color-token file (`ui/theme.py`) —
every text color is paired with a specific background color so nothing
can silently share a color with the surface behind it (a common cause
of "invisible" text in dashboards). Badges, status pills, table text,
and chart labels all reference these tokens rather than relying on
Streamlit's default (light-on-dark can invert unexpectedly depending
on theme).
