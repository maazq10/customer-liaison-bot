from text_to_attribute_I0_data_sources import *
from text_to_attribute_I0 import *

def predict(upc, description, brand=False, subcategory=False, attribute=False):
    if subcategory or attribute:
        return categorize_attribute(upc, description, subcategory, attribute)
    else:
        return process_data(upc, description, brand)
