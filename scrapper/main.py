import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import json


def clean_value(text):
    text = text.replace('იპოვე მსგავსი', '').strip()
    text = ' '.join(text.split())
    return text


def get_product_details(url):
    try:
        print("\n" + "=" * 50)
        print(f"Processing URL: {url}")
        time.sleep(1)

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: Status code {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        code_element = soup.find('span', {'id': lambda x: x and x.startswith('product_code_')})
        code = code_element.text.strip() if code_element else ''
        print(f"\nFound Product Code: {code}")

        features = {}
        feature_elements = soup.find_all('div', class_='ty-product-feature')

        print("\nFound Features:")
        print("-" * 30)

        for feature in feature_elements:
            label_elem = feature.find('div', class_='ty-product-feature__label')
            value_elem = feature.find('div', class_='ty-product-feature__value')

            if label_elem and value_elem:
                label = label_elem.text.strip().replace(':', '')
                value = clean_value(value_elem.text.strip())
                if value:
                    features[label] = value
                    print(f"{label}: {value}")

        if not features:
            print("No features found!")

        print("-" * 30)

        return {
            'code': code,
            'features': features
        }
    except Exception as e:
        print(f"\nError processing {url}")
        print(f"Error details: {str(e)}")
        return None


def process_products(input_csv, output_csv):
    print("\nStarting to process products...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} products from CSV")

    df['code'] = ''
    df['features'] = ''

    for index, row in df.iterrows():
        print(f"\nProcessing product {index + 1}/{len(df)}")
        print(f"Product name: {row['Name']}")

        details = get_product_details(row['Product_URL'])
        if details:
            df.at[index, 'code'] = details['code']
            df.at[index, 'features'] = json.dumps(details['features'], ensure_ascii=False)
            print("\nSuccessfully processed product")
            print(f"Code: {details['code']}")
            print(f"Features count: {len(details['features'])}")
        else:
            print("\nFailed to process product")

    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nScraping complete. Results saved to {output_csv}")


if __name__ == "__main__":
    input_file = "gorgia.csv"
    output_file = "gorgia_products_with_details.csv"
    process_products(input_file, output_file)