import json
import os

import numpy as np
import openai
from fuzzywuzzy import process
from openai import OpenAI
from google.cloud import secretmanager

from text_to_attribute_I0_data_sources import (
    get_attribute_subcategory,
    get_distinct_brands,
    get_distinct_values,
    load_json_company_data,
    tiktoken_len,
    get_openai_api_key
)

project_id = "shining-landing-763"
dataset_id = "STANDARD_MIS_INPUT"
table_id = "common_dim_t_products"

openai_api_key = get_openai_api_key()
client = OpenAI(api_key=openai_api_key)

# Load the JSON data once
json_data = load_json_company_data()

# Get distinct values for department, category, and subcategory
distinct_values = get_distinct_values(project_id, dataset_id, table_id)
distinct_values_dict = json.loads(distinct_values)


def get_confidence(logprob_lst):
    """
    Converts a list of token-probability pairs into a confidence value.
    """
    # sum all the probabilities and disregard the tokens.
    log_probability = np.sum([lp.logprob for lp in logprob_lst.content])

    # convert log-probabilities into probabilities
    confidence = np.round(np.exp(log_probability), 5)
    return confidence


def product_hierarchy(upc, description, distinct_values_dict):
    """
    This function uses the OpenAI GPT-4 model to classify a product into a subcategory
    from the product's description. It then matches the subcategory to a category and department.

    Args:
        upc (str): The UPC of the product.
        description (str): The description of the product.
        distinct_values_dict (list of dict): A list of dictionaries, each containing 'SUBCATEGORY', 'CATEGORY',
        and 'GRP' keys.

    Returns:
        dict: A dictionary containing the UPC, description, department, category, subcategory of the product.

    Raises:
        JSONDecodeError: If the response from the OpenAI API cannot be decoded.
    """
    category = None
    grp = None
    # SUBCATEGORY PROMPT
    subcategories = set(d["SUBCATEGORY"] for d in distinct_values_dict)
    prompt_combined = f"""
    You are a grocery store data analyst. Given this product's description '{description}':
    1. Classify it under the correct grocery or drug store SUBCATEGORY: {subcategories}
    2. Do not include an explanation of your answers.
    3  If you cannot answer the questions respond with 'no prediction'
    Return output in the following JSON format: {{"subcategory": ""}}
    """

    # Count the number of tokens in the prompt
    token_count = tiktoken_len(prompt_combined)

    response_combined = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_combined}],
        max_tokens=60,
        n=1,
        stop=None,
        logprobs=True
        # temperature=0,
    )

    combined = response_combined.choices[0].message.content
    try:
        combined = json.loads(combined)
        subcategory = combined["subcategory"]
    except json.JSONDecodeError:
        print(f"Failed to decode JSON for combined response: {combined}")
        subcategory = None

    # Calculate the confidence of the prediction
    logprob_lst = response_combined.choices[0].logprobs
    subcat_confidence = get_confidence(logprob_lst)

    # MATCH THE SUBCATEGORY TO THE CATEGORY AND DEPARTMENT (GRP IN THE BQ TABLE)
    for item in distinct_values_dict:
        if item["SUBCATEGORY"] == subcategory:
            category = item["CATEGORY"]
            grp = item["GRP"]

    return {
        "upc": upc,
        "description": description,
        "department": grp,
        "category": category,
        "subcategory": subcategory,
        "subcat_confidence": subcat_confidence,
        "hierarchy_token_count": token_count,  # Add the token count to the results
    }


# LOOKUP PROBABLE BRAND BEFORE CLASSIFICATION
def lookup_probable_brand(product_info, json_data):
    """
    This function looks up a probable brand for a product based on the first eight digits of its UPC.
    If the first eight digits exist in the provided JSON data and the 'probable_brand_pct' is 0.9,
    the function assigns the 'probable_brand' from the JSON data to the product info.

    Args:
        product_info (dict): A dictionary containing product information. Must include a 'upc' key.
        json_data (dict): A dictionary where keys are the first eight digits of a UPC and values are
                          dictionaries with 'probable_brand' and 'probable_brand_pct' keys.

    Returns:
        dict: The updated product_info dictionary. If a probable brand was found, it includes a 'brand' key.
    """
    upc = product_info["upc"]
    first_eight_digits = upc[:8]

    # Check if the first eight digits exist in the JSON data and probable_brand_pct is 1
    if (
        first_eight_digits in json_data
        and json_data[first_eight_digits]["probable_brand_pct"] == 0.9
    ):  # TODO - change to 0.9
        probable_brand = json_data[first_eight_digits]["probable_brand"]
        product_info["brand"] = probable_brand

    return product_info


# MATCH SPINS BRANDS AFTER CLASSIFICATION
def match_spins_brands(product_info, project_id, dataset_id, table_id, threshold=70):
    """
    This function matches the brand of a product to the most similar brand in a specific subcategory using fuzzy matching.
    If the match score is above a certain threshold, the function updates the brand in the product info.

    Args:
        product_info (dict): A dictionary containing product information. Must include 'brand', 'department', 'category', and 'subcategory' keys.
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str): The ID of the dataset in BigQuery.
        table_id (str): The ID of the table in BigQuery.
        threshold (int, optional): The minimum match score to consider a brand as a match. Defaults to 70.

    Returns:
        dict: The updated product_info dictionary. If a match was found, the 'brand' key is updated.
    """

    brand = product_info.get("brand")

    # Get distinct brands for the specific subcategory
    distinct_brands = get_distinct_brands(
        project_id,
        dataset_id,
        table_id,
        product_info["department"],
        product_info["category"],
        product_info["subcategory"],
    )
    distinct_brands_dict = json.loads(distinct_brands)
    distinct_brands_list = [d["distinct_value"] for d in distinct_brands_dict]

    # If there are no distinct brands, return the product_info without making any changes
    if not distinct_brands_list:
        return product_info

    # Use fuzzy matching to find the best match for the brand
    best_match, match_score = process.extractOne(brand, distinct_brands_list)

    # If the match score is above the threshold, update the brand
    if match_score >= threshold:
        product_info["brand"] = best_match

    return product_info


# BRAND CLASSIFICATION
def brand_classification(product_info, retailer_brand=None):
    """
    This function uses the OpenAI GPT-4 model to classify a product under the correct grocery brand based on its UPC,
    description, and retailer brand. If a brand has already been determined, the function returns the product info
    without making any changes. If a brand is predicted by the model, the function updates the 'brand' key in the
    product info.

    Args:
        product_info (dict): A dictionary containing product information. Must include 'upc' and 'description' keys.
        retailer_brand (str, optional): The retailer brand of the product. Defaults to None.

    Returns:
        dict: The updated product_info dictionary. If a brand was predicted, it includes a 'brand' key.

    Raises:
        JSONDecodeError: If the response from the OpenAI API cannot be decoded.
    """

    # If a brand has already been determined, don't process the item
    if "brand" in product_info and product_info["brand"] is not None:
        return product_info

    # Get the UPC and description from the product_info dictionary
    upc = product_info["upc"]
    description = product_info["description"]

    # Check if retailer_brand is available
    if retailer_brand:
        prompt_brand = f"""
        You are a grocery store data analyst. Given this product's UPC '{upc}', description '{description}', and retailer brand '{retailer_brand}', classify it under the correct grocery BRAND.

        The product description or retailer brand might contain the brand name. Brand names are often proper nouns and may be located at the beginning of the description or retailer brand.

        Use the following guidelines:

        1. If there is an explicit mention of a brand in the description or retailer brand, consider it the correct brand.
        2. If the description or retailer brand contains an abbreviation or acronym that likely represents a grocery brand, write out the brand name associated with the abbreviation. Do not use the abbreviation answer if it differs from the provided brand.
        3. If the description and retailer brand do not contain any brand information, specify "No prediction".
        4. Only provide the brand prediction, do not include any extra text with your prediction.
        5. Do not explain why there is no prediction.

        Return output in the following JSON format: {{"brand": ""}}"""

    else:
        prompt_brand = f"""
        You are a grocery store data analyst. Given this product's UPC '{upc}' and description '{description}', classify it under the correct grocery BRAND.
                The product description or retailer brand might contain the brand name. Brand names are often proper nouns and may be located at the beginning of the description or retailer brand.

        Use the following guidelines:

        1. If there is an explicit mention of a brand in the description or retailer brand, consider it the correct brand.
        2. If the description or retailer brand contains an abbreviation or acronym that likely represents a grocery brand, write out the brand name associated with the abbreviation. Do not use the abbreviation answer if it differs from the provided brand.
        3. If the description and retailer brand do not contain any brand information, specify "No prediction".
        4. Only provide the brand prediction, do not include any extra text with your prediction.
        5. Do not explain why there is no prediction.

        Return output in the following JSON format: {{"brand": ""}}"""

    # Count the number of tokens in the prompt
    token_count = tiktoken_len(prompt_brand)

    response_brand = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_brand}],
        max_tokens=60,
        n=1,
        stop=None,
        logprobs=True,
    )
    brand_response = response_brand.choices[0].message.content
    try:
        brand_response = json.loads(brand_response)
        if "brand" in brand_response and brand_response["brand"]:
            product_info["brand"] = brand_response["brand"]
    except json.JSONDecodeError:
        print(f"Failed to decode JSON for brand response: {brand_response}")

    # Calculate the confidence of the prediction
    logprob_lst = response_brand.choices[0].logprobs
    brand_confidence = get_confidence(logprob_lst)
    product_info["brand_confidence"] = brand_confidence
    product_info[
        "brand_token_count"
    ] = token_count  # Add the token count to the results

    return product_info


# CATEGORIZE ATTRIBUTE (NO ATTRIBUTE CONTEXT)=
def categorize_attribute(upc, description, subcategory, attribute):
    """
    This function uses the OpenAI API to categorize a product based on a specified attribute.
    It sends a prompt to the OpenAI API for the product, and parses the response
    to get the attribute value. The function then returns the results.

    Args:
        upc (str): The UPC of the product.
        description (str): The description of the product.
        subcategory (str): The subcategory of the product.
        attribute (str): The attribute to categorize the product by.

    Returns:
        dict: A dictionary containing the results. Includes 'upc', 'description', 'subcategory', and the attribute.
    """
    # Get the distinct attribute values for the subcategory from the BigQuery table
    attribute_values_json = get_attribute_subcategory(
        project_id, dataset_id, table_id, attribute, subcategory
    )
    attribute_values_dict = json.loads(attribute_values_json)
    attribute_values = set(
        d[attribute] for d in attribute_values_dict if d[attribute] is not None
    )

    # Use the context in your prompt
    prompt = f"""
    Given this product's UPC '{upc}', description '{description}', and subcategory '{subcategory}':
    1. Classify it under the correct {attribute}: {attribute_values}
    2. If you cannot answer the question respond with 'no prediction'
    Return output in the following format: {{"{attribute}": ""}}
    """
    # Count the number of tokens in the prompt
    token_count = tiktoken_len(prompt)

    # Send the prompt to the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        n=1,
        stop=None,
        logprobs=True,
    )

    # Parse the response
    response_content = response.choices[0].message.content
    try:
        response_dict = json.loads(response_content)
        attribute_value = response_dict[attribute]
    except json.JSONDecodeError:
        print(f"Failed to decode JSON for response: {response_content}")
        attribute_value = "No prediction"

    # Calculate the confidence of the prediction
    logprob_lst = response.choices[0].logprobs
    confidence = get_confidence(logprob_lst)

    # Add the result to the list
    result = {
        "upc": upc,
        "description": description,
        "subcategory": subcategory,
        "token_count": token_count,  # Add the token count to the results
        attribute: attribute_value,
        f"{attribute}_confidence": confidence,  # Add the confidence to the results
    }

    return result


# PROCESS HIERARCHY AND BRAND DATA
def process_data(upc, description, brand=False):
    """
    This function processes a product data, classifying the product into a hierarchy and/or matching or classifying its brand.
    The final results are returned as a JSON object.

    Args:
        upc (str): The UPC of the product.
        description (str): The description of the product.
        brand (bool, optional): If True, match or classify the product's brand. Defaults to False.

    Returns:
        str: A JSON object containing the processed product data.
    """
    product_info = {"upc": upc, "description": description}
    product_info = product_hierarchy(
        upc, description, distinct_values_dict
    )  # Pass distinct_values_dict as an argument

    if brand:
        product_info = lookup_probable_brand(product_info, json_data)
        product_info = brand_classification(product_info)
        result = match_spins_brands(product_info, project_id, dataset_id, table_id)
        product_info = result

    return product_info
