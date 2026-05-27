# KOKORO — Japanese Brand Sentiment Engine

A sentiment analysis tool for monitoring how Japanese consumers talk about brands and products across major Japanese social media and review platforms. Scrapes comments in real time, classifies them as Positive / Neutral / Negative using a Japanese NLP model, and displays the results in a dark-themed Streamlit dashboard.

---

## Features

- **5 data sources** — Yahoo Japan News, @cosme, Tabelog, Kakaku.com, Twitter / X
- **Japanese NLP** — powered by a Transformer model fine-tuned on Japanese review data
- **Live dashboard** — sentiment breakdown, trend charts, rolling 7-day average, week-over-week alerts
- **Daily monitoring** — run repeatedly; new comments are deduplicated and appended automatically
- **Brand comparison** — compare sentiment across multiple keywords side by side
- **Event annotations** — mark product launches or campaigns on the trend chart

---

## Project Structure

```
SentimentProject/
├── src/
│   ├── dashboard.py          # Streamlit app (landing page + dashboard)
│   ├── analyse.py            # Scrape + classify pipeline (CLI entry point)
│   ├── sentiment.py          # Sentiment model wrapper
│   ├── scraper.py            # Yahoo Japan News scraper
│   ├── scraper_cosme.py      # @cosme beauty reviews scraper
│   ├── scraper_tabelog.py    # Tabelog restaurant reviews scraper
│   ├── scraper_kakaku.py     # Kakaku.com product reviews scraper
│   ├── scraper_5ch.py        # 5ch (2channel) scraper
│   ├── scraper_twitter.py    # Twitter / X scraper (requires API key)
│   └── label.py              # Manual labelling helper
├── models/
│   ├── finetune.py           # Fine-tuning script
│   └── benchmark.py          # Model benchmarking script
├── data/
│   ├── labelled_comments.csv # Labelled training data
│   └── sample_posts.csv      # Sample data for testing
├── .streamlit/
│   └── config.toml           # Dark theme configuration
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Berfil/SentimentAnalyzer.git
cd SentimentAnalyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright browsers (needed for web scrapers)

```bash
playwright install chromium
```

---

## Running the Dashboard

```bash
streamlit run src/dashboard.py
```

The app opens at `http://localhost:8501`. You land on the KOKORO home page — click **Launch Dashboard** to enter the tool.

---

## Using the Dashboard

### Running a new analysis

1. Enter a **keyword** in the sidebar (e.g. `資生堂`, `任天堂`, `スタバ`)
2. Select a **source** (Yahoo Japan News, @cosme, Tabelog, Kakaku.com, Twitter, or All)
3. Set the number of articles / items to scrape
4. Click **▶ Run Analysis**

The scraper fetches comments, runs sentiment classification, and displays the results. Results are saved to `data/scraped_comments.csv` and appended on subsequent runs (duplicates are filtered automatically).

### Loading existing data

- Click **Load default CSV** in the sidebar to load a previously saved `scraped_comments.csv`
- Or drag and drop any CSV file exported from a previous run into the upload area on the empty state screen

### Twitter / X

Select **Twitter / X** as the source and paste your **Bearer Token** from the [Twitter Developer Portal](https://developer.twitter.com) into the sidebar field. Requires a Basic tier or higher API plan.

---

## Running the Scraper from the Command Line

You can also run the analysis pipeline directly without the dashboard:

```bash
python src/analyse.py \
  --keyword "資生堂" \
  --source yahoo \
  --max_articles 20 \
  --output data/scraped_comments.csv
```

**Sources:** `yahoo`, `cosme`, `tabelog`, `kakaku`, `twitter`, `all`

For Twitter:
```bash
python src/analyse.py \
  --keyword "任天堂" \
  --source twitter \
  --max_tweets 100 \
  --bearer_token YOUR_TOKEN \
  --output data/scraped_comments.csv
```

---

## Daily Monitoring Workflow

To track sentiment over time, run the tool each morning with the same keyword and output file. New comments are automatically deduplicated — only genuinely new content is added. Each row is stamped with the `keyword` and `scraped_date` columns so you can filter by date in the dashboard.

Example cron job (runs every day at 8 AM):
```
0 8 * * * cd /path/to/SentimentAnalyzer && .venv/bin/python src/analyse.py --keyword "資生堂" --source all --output data/scraped_comments.csv
```

---

## Model

The sentiment engine uses a custom fine-tuned model built on top of [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment), fine-tuned by **Filip Berndtsson** on a hand-labelled dataset of Japanese social media comments (`data/labelled_comments.csv`).

Fine-tuning was done specifically for the Japanese market — the base model was trained on multilingual Twitter data, but the fine-tuning step adapts it to the tone, vocabulary, and writing style of Japanese review platforms and news comment sections.

The fine-tuned weights are stored locally at `models/finetuned_cardiffnlp/best/` and are excluded from the repository (too large for git). If the weights are not present, re-train with:

```bash
python models/finetune.py
```

Labelled training data lives in `data/labelled_comments.csv` (columns: `comment`, `sentiment`). Adding more labelled examples and re-running the script will improve accuracy.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| Chromium | Installed via `playwright install chromium` |
| Twitter Bearer Token | Only needed for Twitter / X source |
| ~4 GB RAM | For loading the NLP model |
| GPU (optional) | Speeds up classification significantly |
