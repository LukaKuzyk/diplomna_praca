"""
Tickers Configuration for ML Finance Project.
Contains 30 Large-Cap companies + 1 Benchmark (DIA).
Defines Name, Sector, and 3-5 specific Google Trends Keywords for Search and News.
"""

TICKER_CONFIG = {
    "AAPL": {
        "name": "Apple",
        "sector": "Information Technology",
        "search_kw": ["iPhone", "Apple AI", "stock"],
        "news_kw": ["Tech earnings", "Apple news", "tariffs"]
    },
    "MSFT": {
        "name": "Microsoft",
        "sector": "Information Technology",
        "search_kw": ["Windows", "Azure", "ChatGPT"],
        "news_kw": ["Microsoft earnings", "AI news", "tech sector"]
    },
    "NVDA": {
        "name": "Nvidia",
        "sector": "Information Technology",
        "search_kw": ["GPU", "Nvidia AI", "graphics card"],
        "news_kw": ["Semiconductor news", "AI chips", "tech earnings"]
    },
    "GOOGL": {
        "name": "Alphabet",
        "sector": "Communication Services",
        "search_kw": ["Google AI", "Gemini", "YouTube"],
        "news_kw": ["Ad revenue", "Tech antitrust", "earnings"]
    },
    "META": {
        "name": "Meta Platforms",
        "sector": "Communication Services",
        "search_kw": ["Instagram", "Facebook", "VR"],
        "news_kw": ["Social media regulations", "Meta news", "ad revenue"]
    },
    "NFLX": {
        "name": "Netflix",
        "sector": "Communication Services",
        "search_kw": ["Netflix movies", "subscription", "Netflix series"],
        "news_kw": ["Streaming wars", "Netflix earnings", "entertainment"]
    },
    "AMZN": {
        "name": "Amazon",
        "sector": "Consumer Discretionary",
        "search_kw": ["Amazon Prime", "AWS", "shopping"],
        "news_kw": ["E-commerce news", "retail sales", "Amazon earnings"]
    },
    "TSLA": {
        "name": "Tesla",
        "sector": "Consumer Discretionary",
        "search_kw": ["Tesla Model 3", "Elon Musk", "EV"],
        "news_kw": ["Electric vehicles", "Tesla news", "car sales"]
    },
    "HD": {
        "name": "Home Depot",
        "sector": "Consumer Discretionary",
        "search_kw": ["Home improvement", "DIY", "hardware"],
        "news_kw": ["Housing market", "retail earnings", "construction"]
    },
    "PG": {
        "name": "Procter & Gamble",
        "sector": "Consumer Staples",
        "search_kw": ["P&G products", "Gillette", "Tide"],
        "news_kw": ["Consumer spending", "inflation", "P&G earnings"]
    },
    "KO": {
        "name": "Coca-Cola",
        "sector": "Consumer Staples",
        "search_kw": ["Coca-Cola", "Sprite", "drinks"],
        "news_kw": ["Beverage industry", "inflation", "consumer goods"]
    },
    "WMT": {
        "name": "Walmart",
        "sector": "Consumer Staples",
        "search_kw": ["Walmart groceries", "shopping", "discount"],
        "news_kw": ["Retail earnings", "supply chain", "consumer spending"]
    },
    "JNJ": {
        "name": "Johnson & Johnson",
        "sector": "Health Care",
        "search_kw": ["Tylenol", "J&J skincare", "medicine"],
        "news_kw": ["Pharma news", "FDA approval", "healthcare"]
    },
    "UNH": {
        "name": "UnitedHealth",
        "sector": "Health Care",
        "search_kw": ["Health insurance", "Medicare", "health plan"],
        "news_kw": ["Healthcare policy", "insurance news", "health industry"]
    },
    "PFE": {
        "name": "Pfizer",
        "sector": "Health Care",
        "search_kw": ["Pfizer vaccine", "medicine", "pharmacy"],
        "news_kw": ["Clinical trials", "FDA approval", "pharma earnings"]
    },
    "JPM": {
        "name": "JPMorgan Chase",
        "sector": "Financials",
        "search_kw": ["Chase bank", "credit card", "loan"],
        "news_kw": ["Interest rates", "banking news", "economy"]
    },
    "V": {
        "name": "Visa",
        "sector": "Financials",
        "search_kw": ["Visa card", "credit card", "payments"],
        "news_kw": ["Consumer spending", "inflation", "financial sector"]
    },
    "BAC": {
        "name": "Bank of America",
        "sector": "Financials",
        "search_kw": ["BofA login", "mortgage", "banking"],
        "news_kw": ["Interest rates", "Federal Reserve", "banks"]
    },
    "CAT": {
        "name": "Caterpillar",
        "sector": "Industrials",
        "search_kw": ["Caterpillar equipment", "tractor", "machinery"],
        "news_kw": ["Construction index", "infrastructure", "industrial news"]
    },
    "BA": {
        "name": "Boeing",
        "sector": "Industrials",
        "search_kw": ["Boeing 737", "flight tickets", "aviation"],
        "news_kw": ["Aviation news", "defense contracts", "aerospace"]
    },
    "LMT": {
        "name": "Lockheed Martin",
        "sector": "Industrials",
        "search_kw": ["Fighter jet", "F-35", "defense"],
        "news_kw": ["Defense budget", "war news", "military contracts"]
    },
    "XOM": {
        "name": "Exxon Mobil",
        "sector": "Energy",
        "search_kw": ["Gas prices", "Exxon fuel", "oil"],
        "news_kw": ["Oil prices", "OPEC news", "energy sector"]
    },
    "CVX": {
        "name": "Chevron",
        "sector": "Energy",
        "search_kw": ["Chevron gas", "oil prices", "fuel"],
        "news_kw": ["Oil industry", "energy policy", "OPEC"]
    },
    "COP": {
        "name": "ConocoPhillips",
        "sector": "Energy",
        "search_kw": ["Gas prices", "crude oil", "energy"],
        "news_kw": ["Oil production", "OPEC news", "energy earnings"]
    },
    "LIN": {
        "name": "Linde",
        "sector": "Materials",
        "search_kw": ["Industrial gas", "hydrogen", "materials"],
        "news_kw": ["Chemical industry", "raw materials", "manufacturing"]
    },
    "SHW": {
        "name": "Sherwin-Williams",
        "sector": "Materials",
        "search_kw": ["Paint", "home renovation", "colors"],
        "news_kw": ["Construction news", "housing market", "materials sector"]
    },
    "PLD": {
        "name": "Prologis",
        "sector": "Real Estate",
        "search_kw": ["Warehouses", "logistics", "storage"],
        "news_kw": ["Real estate news", "interest rates", "supply chain"]
    },
    "O": {
        "name": "Realty Income",
        "sector": "Real Estate",
        "search_kw": ["Dividend", "REIT", "rental"],
        "news_kw": ["Commercial real estate", "interest rates", "real estate market"]
    },
    "NEE": {
        "name": "NextEra Energy",
        "sector": "Utilities",
        "search_kw": ["Solar power", "electricity", "wind energy"],
        "news_kw": ["Renewable energy", "utilities news", "energy grid"]
    },
    "DUK": {
        "name": "Duke Energy",
        "sector": "Utilities",
        "search_kw": ["Power outage", "electricity bill", "energy"],
        "news_kw": ["Energy grid", "utilities regulations", "power sector"]
    },
    "DIA": {
        "name": "Dow Jones ETF",
        "sector": "Benchmark",
        "search_kw": ["Dow Jones", "stock market", "investing"],
        "news_kw": ["US economy", "recession", "Federal Reserve"]
    }
}
