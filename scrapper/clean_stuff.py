import pandas as pd
import json


def clean_csv(input_file, output_file):
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Initial number of rows: {initial_count}")
    df = df[df['code'].notna() & (df['code'] != '')]
    after_code_count = len(df)
    print(f"Rows after removing empty codes: {after_code_count}")
    print(f"Removed {initial_count - after_code_count} rows with empty codes")

    valid_features = []
    for idx, row in df.iterrows():
        try:
            features = json.loads(row['features'])
            if features:
                valid_features.append(True)
            else:
                valid_features.append(False)
        except:
            valid_features.append(False)

    df = df[valid_features]
    final_count = len(df)
    print(f"Final number of rows: {final_count}")
    print(f"Removed {after_code_count - final_count} rows with invalid features")
    df.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to: {output_file}")
    print(f"Total rows removed: {initial_count - final_count}")
    print(f"Percentage of data kept: {(final_count / initial_count * 100):.2f}%")


if __name__ == "__main__":
    input_file = "gorgia_products_with_details.csv"
    output_file = "gorgia_products_cleaned.csv"
    clean_csv(input_file, output_file)