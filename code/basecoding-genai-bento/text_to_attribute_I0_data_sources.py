import json
import os
import pandas as pd

import dotenv
import tiktoken
import numpy as np
dotenv.load_dotenv('env', override=True)

from google.cloud import bigquery
from google.cloud import storage

from pydantic import BaseModel, Field
from google.cloud import secretmanager

def get_openai_api_key():

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/spins-retail-solutions/secrets/openai-api-key/versions/latest"
    # Access the secret version.
    response = client.access_secret_version(request={"name": secret_name})

    # WARNING: Do not print the secret in a production environment - this snippet
    # is showing how to access the secret material.
    secret = response.payload.data.decode("UTF-8")
    return secret


openai_api_key = os.environ["OPENAI_API_KEY"]

def get_prediction_data():
    """
    This function retrieves the most recent product data from the Google Cloud Storage bucket for the existing Base Coding models,
    downloads the ntc released upc items, and reads it into a pandas DataFrame. The function filters the blobs in the bucket based 
    on the specific time period release prefix and sorts them in descending order to find the most recent blob. It then downloads 
    this blob to a CSV file and reads specific columns into a DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the product data. Includes 'upc', 'retailer_brand', 'retailer_item_description', 
        'retailer_unit_size', and 'retailer_unit_of_measure' columns.
    """
    # Create a storage client
    storage_client = storage.Client(project="spins-retail-solutions")

    # Get the bucket
    bucket = storage_client.get_bucket('product-intelligence')

    # List all blobs in the bucket
    blobs = bucket.list_blobs(prefix='data/')

    # Filter the blobs based on the prefix and sort them in descending order
    csv_blobs = sorted([blob for blob in blobs if blob.name.endswith('ntc_released_upcs.csv') and 'released_data' in blob.name], key=lambda blob: blob.name, reverse=True)

    # Get the most recent blob
    most_recent_blob = csv_blobs[0]

    # Download the blob to a CSV file
    most_recent_blob.download_to_filename('/tmp/most_recent_data.csv')

    # Specify the columns you want to import
    columns_to_import = ['upc','retailer_brand','retailer_item_description','retailer_unit_size','retailer_unit_of_measure']

    # Use pandas to read the CSV file with specific columns
    df = pd.read_csv('/tmp/most_recent_data.csv', usecols=columns_to_import, dtype=str)

    return df

project_id = 'shining-landing-763'
dataset_id = 'STANDARD_MIS_INPUT'
table_id = 'common_dim_t_products'

def get_distinct_values(project_id, dataset_id, table_id):
    """
    This function retrieves distinct values of 'GRP', 'CATEGORY', and 'SUBCATEGORY' from a specified BigQuery table.
    It executes a SQL query to fetch the distinct values, stores the results in a dictionary, and then converts the 
    dictionary to a JSON string.

    Args:
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str): The ID of the dataset in BigQuery.
        table_id (str): The ID of the table in BigQuery.

    Returns:
        str: A JSON string containing the distinct values. Each item in the JSON string is a dictionary with 'GRP', 
        'CATEGORY', and 'SUBCATEGORY' keys.
    """

    client = bigquery.Client(project="spins-retail-solutions") 

    query = f"""
        SELECT DISTINCT GRP, CATEGORY, SUBCATEGORY
        FROM `{project_id}.{dataset_id}.{table_id}`
    """
    
    # Execute the query
    query_job = client.query(query)
    
    # Fetch the results
    results = query_job.result()
    
    # Store the results in a dictionary
    results_dict = [{'GRP': row['GRP'], 'CATEGORY': row['CATEGORY'], 'SUBCATEGORY': row['SUBCATEGORY']} for row in results]

    # Convert the results dictionary to a JSON string
    results_json = json.dumps(results_dict) if results_dict else json.dumps({})

    # Return the results
    return results_json

def get_attribute_subcategory(project_id, dataset_id, table_id, attribute=None, subcategory=None):
    """
    This function retrieves distinct values of 'GRP', 'CATEGORY', 'SUBCATEGORY', and an optional attribute from a specified BigQuery table.
    It executes a SQL query to fetch the distinct values, stores the results in a dictionary, and then converts the 
    dictionary to a JSON string. The function can also filter the results by a specific subcategory.

    Args:
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str): The ID of the dataset in BigQuery.
        table_id (str): The ID of the table in BigQuery.
        attribute (str, optional): An additional attribute to include in the query. If None, no additional attribute is included. Defaults to None.
        subcategory (str, optional): The subcategory to filter by. If None, no subcategory filter is applied. Defaults to None.

    Returns:
        str: A JSON string containing the distinct values. Each item in the JSON string is a dictionary with 'GRP', 
        'CATEGORY', 'SUBCATEGORY', and the optional attribute keys.
    """
        
    client = bigquery.Client(project="spins-retail-solutions") 

    # Include the attribute in the query if it is specified
    attribute_query = f", {attribute}" if attribute else ""
    
    # Include the subcategory in the query if it is specified
    subcategory_query = f" WHERE SUBCATEGORY = '{subcategory}'" if subcategory else ""
    
    query = f"""
        SELECT DISTINCT GRP, CATEGORY, SUBCATEGORY{attribute_query}
        FROM `{project_id}.{dataset_id}.{table_id}`{subcategory_query}
    """
    
    # Execute the query
    query_job = client.query(query)
    
    # Fetch the results
    results = query_job.result()
    
    # Include the attribute in the dictionary if it is specified
    results_dict = [{'GRP': row['GRP'], 'CATEGORY': row['CATEGORY'], 'SUBCATEGORY': row['SUBCATEGORY'], attribute: row[attribute] if attribute else None} for row in results]

    # Convert the results dictionary to a JSON string
    results_json = json.dumps(results_dict) if results_dict else json.dumps({})

    # Return the results
    return results_json


def get_distinct_brands(project_id, dataset_id, table_id, department, category, subcategory):
    """
    This function retrieves distinct brands from a specified BigQuery table for a given department, category, and subcategory.
    It executes a SQL query to fetch the distinct brands and stores the results in a list, and then converts the 
    list to a JSON string.

    Args:
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str): The ID of the dataset in BigQuery.
        table_id (str): The ID of the table in BigQuery.
        department (str): The department to filter by.
        category (str): The category to filter by.
        subcategory (str): The subcategory to filter by.

    Returns:
        str: A JSON string containing the distinct brands. Each item in the JSON string is a dictionary with a 'distinct_value' key.
    """

    client = bigquery.Client(project="spins-retail-solutions") 

    query = f"""
        SELECT DISTINCT BRAND as distinct_value
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE GRP = '{department}' AND CATEGORY = '{category}' AND SUBCATEGORY = '{subcategory}'
        GROUP BY BRAND
    """
    
    # Execute the query
    query_job = client.query(query)
    
    # Fetch the results
    results = query_job.result()
    
    # Store the results in a list
    results_list = [{ 'distinct_value': row['distinct_value']} for row in results]

    # Convert the results to a JSON string
    results_json = json.dumps(results_list) if results_list else json.dumps([])

    # Return the results
    return results_json

def load_json_company_data():
    """
    This function loads JSON company data from a blob in a Google Cloud Storage bucket. 
    It creates a storage client, gets the bucket and the blob, and tries to download the blob to a JSON file and load the JSON data. 
    If an error occurs during the download or loading, it retries up to 3 times.

    Returns:
        dict: A dictionary containing the JSON data. If an error occurs during the download or loading, an empty dictionary is returned.
    """
        
    # Create a client
    storage_client = storage.Client(project="spins-retail-solutions")

    # Get the bucket
    bucket = storage_client.get_bucket('product-intelligence')

    # Get the blob
    blob = bucket.blob('data/period=latest/upc_brand_matching_dict.json')

    # Initialize json_data as an empty dictionary
    json_data = {}

    # Try to download the blob to a JSON file and load the JSON data
    for _ in range(3):  # Retry up to 3 times
        try:
            # Download the blob to a JSON file
            blob.download_to_filename('/tmp/latest_upc_brand_matching_dict.json')

            # Load the JSON data from the file
            with open('/tmp/latest_upc_brand_matching_dict.json', "r") as json_file:
                json_data = json.load(json_file)
            break  # If the download and loading were successful, break the loop
        except (json.JSONDecodeError, google.cloud.exceptions.NotFound) as e:
            print(f"Error: {e}")

    return json_data

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
 
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)
