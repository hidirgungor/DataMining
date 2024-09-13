import requests
from bs4 import BeautifulSoup

def fetch_website_content(url):
    try:
        # Send an HTTP request to the website
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.content
        else:
            print(f"Failed to load the page. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def scrape_content(html_content, output_file):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Open file for writing
    with open(output_file, 'w', encoding='utf-8') as f:
        # Extract headings (h1, h2, h3, etc.)
        f.write("HEADINGS:\n")
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            f.write(header.get_text() + "\n")
        
        # Extract paragraphs
        f.write("\nPARAGRAPHS:\n")
        for paragraph in soup.find_all('p'):
            f.write(paragraph.get_text() + "\n")
        
        # Extract images (img tags)
        f.write("\nIMAGES:\n")
        for img in soup.find_all('img'):
            img_src = img.get('src')
            if img_src:
                f.write(img_src + "\n")
        
        # Extract links (a tags)
        f.write("\nLINKS:\n")
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                f.write(href + "\n")
    
    print(f"Scraped content has been saved to {output_file}")

def main():
    # Get the URL from the user
    url = input("Please enter the website URL to scrape: ")
    
    # Fetch the page content
    content = fetch_website_content(url)
    
    if content:
        # Specify output file name
        output_file = 'scraped_content.txt'
        # Scrape the content and save it to file
        scrape_content(content, output_file)

if __name__ == "__main__":
    main()
