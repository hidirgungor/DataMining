import pandas as pd
import numpy as np
import requests

# Using requests to read the web page
url = "https://www.randomtextgenerator.com/"
response = requests.get(url)

# Reading the HTML content from the page
text = response.content.decode('utf-8')

# Cleaning the data
clean_text = text.replace('\n', '').replace('\t', '').replace('\r', '').replace('&nbsp;', '').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&cent;', '¢').replace('&pound;', '£').replace('&yen;', '¥').replace('&euro;', '€').replace('&copy;', '©').replace('&reg;', '®').replace('&trade;', '™').replace('&times;', '×').replace('&divide;', '÷')

# Converting the text into a Pandas dataframe
df = pd.read_html(clean_text)

# Dropping null values
df.dropna(inplace=True)

# Selecting a random sample from the dataframe
sample = df.sample()

# Printing the sample
print(sample)
