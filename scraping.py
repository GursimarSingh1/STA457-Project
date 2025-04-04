import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import re
import unicodedata
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def generate_date_urls(start_date, end_date):
    """Generate URLs for each day between start_date and end_date."""
    urls = []
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        url = f"https://www.ghanaweb.com/GhanaHomePage/business/browse.archive.php?date={date_str}"
        urls.append((date_str, url))
        current_date += timedelta(days=1)
    
    return urls

def get_articles_from_listing_page(url, date_str):
    """Extract all article links and titles from a date listing page."""
    articles = []
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the div with the article listings
        article_container = soup.find('div', class_='left_artl_list more_news')
        
        if article_container:
            # Find all articles in the unordered list
            upper_div = article_container.find('div', class_='upper')
            if upper_div:
                article_list = upper_div.find('ul')
                if article_list:
                    for item in article_list.find_all('li'):
                        link = item.find('a')
                        if link and link.has_attr('href') and link.has_attr('title'):
                            title = link['title']
                            href = link['href']
                            
                            # Check if "cocoa" is in the title (case insensitive)
                            if re.search(r'cocoa', title, re.IGNORECASE):
                                articles.append({
                                    'date': date_str,
                                    'title': title,
                                    'url': href if href.startswith('http') else f"https://www.ghanaweb.com{href}"
                                })
    except Exception as e:
        print(f"Error processing {url}: {e}")
    
    return articles

def get_article_content(article_info):
    """Get the full content of an article."""
    try:
        response = requests.get(article_info['url'])
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the article content - look for p tag with id='article-123'
        article_paragraph = soup.find('p', id='article-123')
        
        if article_paragraph:
            # Extract text content
            content = article_paragraph.get_text(strip=True)
            article_info['content'] = content
        else:
            print(f"No content found for {article_info['url']} - trying alternative selectors")
            
            # Try alternative selectors
            article_div = soup.find('div', class_='article-content-area')
            if article_div:
                paragraphs = article_div.find_all('p')
                if paragraphs:
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                    article_info['content'] = content
                    return article_info
            
            # If we get here, no content was found
            article_info['content'] = "Content not found"
            
    except Exception as e:
        print(f"Error fetching article {article_info['url']}: {e}")
        article_info['content'] = f"Error: {str(e)}"
    
    return article_info

def clean_text(text):
    """
    Comprehensive text cleaning function.
    
    Args:
    text (str): Input text to be cleaned
    
    Returns:
    str: Cleaned text
    """
    # Handle None or non-string inputs
    if not isinstance(text, str):
        return ""
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Trim whitespace
    text = text.strip()
    
    return text

def clean_dataframe_add_sentiment(df):
    """
    Clean text columns in the DataFrame.
    
    Args:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy of the DataFrame to avoid modifying the original
    cleaned_df = df.copy()
    sia = SentimentIntensityAnalyzer()
    
    # Clean title column
    if 'title' in cleaned_df.columns:
        cleaned_df['title'] = cleaned_df['title'].apply(clean_text)
    
    # Clean content column
    if 'content' in cleaned_df.columns:
        cleaned_df['content'] = cleaned_df['content'].apply(clean_text)
    
    # Optional: Remove rows with empty content
    cleaned_df = cleaned_df[cleaned_df['content'].str.strip() != '']
    
    # Reset index after filtering
    cleaned_df = cleaned_df.reset_index(drop=True)

    cleaned_df['sentiment_score'] = cleaned_df['content'].apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )
    
    return cleaned_df

# Define date range (from January 1, 2024 to current date in 2025)
start_date = datetime(2015, 1, 1)
end_date = datetime(2025, 12, 31)  # Using current date

# Generate URLs for each date
date_urls = generate_date_urls(start_date, end_date)
print(f"Generated {len(date_urls)} date URLs to process")

# List to store all cocoa-related articles
all_cocoa_articles = []

# Process each date URL
for i, (date_str, url) in enumerate(date_urls):
    print(f"Processing {date_str} ({i+1}/{len(date_urls)})")
    
    # Get cocoa articles from this date
    cocoa_articles = get_articles_from_listing_page(url, date_str)
    
    if cocoa_articles:
        print(f"Found {len(cocoa_articles)} cocoa-related articles on {date_str}")
        
        # Get content for each article
        for article in cocoa_articles:
            article_with_content = get_article_content(article)
            all_cocoa_articles.append(article_with_content)
            
            # Add a small delay to avoid overwhelming the server
            time.sleep(1)
    
    # Add a delay between date pages
    time.sleep(2)

# Create DataFrame
if all_cocoa_articles:
    df = pd.DataFrame(all_cocoa_articles)
    
    # Convert date string to datetime
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
    
    # Reorder columns
    df = df[['date', 'title', 'content', 'url']]

    df = clean_dataframe_add_sentiment(df)
    
    # Save to CSV
    df[['date', 'sentiment_score']].to_csv(f'sentiment_data.csv', index=False)
    print(f"Saved to sentiment_data.csv")
    
else:
    print("No cocoa-related articles found")