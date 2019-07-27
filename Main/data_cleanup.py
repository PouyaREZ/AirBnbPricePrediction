''' Update notes by PRK Nov 10:
        Taken out zipcode column entirely and added three more column removals (have commented in front)
        Also commented out the print order'''


# Importing the required libraries and methods
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime as dt
# Importing the dataset
filename = '../Data/listings.csv'
reviews_filename = '../Data/reviews_cleaned.csv'
data = pd.read_csv(filename)
reviews = pd.read_csv(reviews_filename, names = ['listing_id', 'comments'])
# print(data.info)
# print(list(data))
# print(list(data)[43])
# print(list(data)[87])
# print(list(data)[88])

# Taking out the unwanted columns
print(len(data.columns))
exit()
data = pd.DataFrame.drop(data, columns=[
    'host_name',
    'notes', # Added PRK
    'host_about', # Added PRK
    'calendar_updated', # Added PRK
    'host_acceptance_rate',
    'description',
    'thumbnail_url',
    'experiences_offered',
    'listing_url',
    'name',
    'summary',
    'space',
    'scrape_id',
    'last_scraped',
    'neighborhood_overview',
    'transit',
    'access',
    'interaction',
    'house_rules',
    'medium_url',
    'picture_url',
    'xl_picture_url',
    'host_url',
    'host_thumbnail_url',
    'host_picture_url',
    'host_acceptance_rate',
    'smart_location',
    'license',
    'jurisdiction_names',
    'street',
    'neighbourhood',
    'country',
    'country_code',
    'host_location',
    'host_neighbourhood',
    'market',
    'is_location_exact',
    'square_feet',
    'weekly_price',
    'monthly_price',
    'availability_30',
    'availability_60',
    'availability_90',
    'availability_365',
    'calendar_last_scraped',
    'first_review',
    'last_review',
    'requires_license',
    'calculated_host_listings_count',
    'host_listings_count',

     #discuss last two
    'zipcode' # Added PRK

])
# print(list(data))


print('Splitting host verifications')
host_verification_set = set()

def collect_host_verifications(entry):
    entry_list = entry.replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace(" ", "").split(',')
    for verification in entry_list:
        if (verification != "" and verification != 'None'):
            host_verification_set.add(verification +"_verification")

data['host_verifications'].apply(collect_host_verifications)

def generic_verification(entry, v):
    entry_list = str(entry).replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace(" ", "").split(',')
    for verification in entry_list:
        if (verification + "_verification" == v):
            return 1
    return 0

for v in host_verification_set:
    data.insert(len(list(data)), v, 0)
    data[v] = data['host_verifications'].apply(lambda x: generic_verification(x, v))

data = pd.DataFrame.drop(data, columns=['host_verifications'])

def clean_response_rate(entry):
    if (type(entry) == str):
        return entry.replace('%', '')
    else:
        return 0


data['host_response_rate'] = data['host_response_rate'].apply(clean_response_rate)

def clean_superhost(entry):
    if (entry == 't'):
        return 1
    else:
        return 0
data['host_is_superhost'] = data['host_is_superhost'].apply(clean_superhost)
data['host_has_profile_pic'] = data['host_has_profile_pic'].apply(clean_superhost)
data['host_identity_verified'] = data['host_identity_verified'].apply(clean_superhost)
data['has_availability'] = data['has_availability'].apply(clean_superhost)
data['instant_bookable'] = data['instant_bookable'].apply(clean_superhost)
data['is_business_travel_ready'] = data['is_business_travel_ready'].apply(clean_superhost)
data['require_guest_profile_picture'] = data['require_guest_profile_picture'].apply(clean_superhost)
data['require_guest_phone_verification'] = data['require_guest_phone_verification'].apply(clean_superhost)

"""
print(list(data))

print(data['host_verifications'][0])
for v in host_verification_set:
    print(v, " ", data[v][0])
"""
def clean_price(entry):
    if (type(entry) != str and math.isnan(entry)):
        return -55
    entry1 = entry.replace('$', '').replace(',', '')
    if (float(entry1) == 0):
        return -55
    return np.log(float(entry1))


def clean_number(entry):
    if (math.isnan(entry)):
        return 0
    else:
        return entry
def clean_number_removal(entry):
    if (math.isnan(entry)):
        return -55
    else:
        return entry
data['bathrooms'] = data['bathrooms'].apply(clean_number_removal)
data['bedrooms'] = data['bedrooms'].apply(clean_number_removal)
data['beds'] = data['beds'].apply(clean_number_removal)
data = data[data['bathrooms'] != -55]
data = data[data['bedrooms'] != -55]
data = data[data['beds'] != -55]

def reviews_per_month_cleanup(entry):
    if (math.isnan(entry)):
        return 0
    return entry

data['reviews_per_month'] = data['reviews_per_month'].apply(reviews_per_month_cleanup)
data['price'] = data['price'].apply(clean_price)
data = data[data['price'] != -55]
data['extra_people'] = data['extra_people'].apply(clean_price)
data['security_deposit'] = data['security_deposit'].apply(clean_price)
data['cleaning_fee'] = data['cleaning_fee'].apply(clean_price)
def clean_listings_count(entry):
    if (math.isnan(entry)):
        return 1
    return entry
data['host_total_listings_count'] = data['host_total_listings_count'].apply(clean_listings_count)
print("Cleaning the state")
def cleaned_state(entry):
    if (isinstance(entry, str)):
        if (entry.upper() == 'NY' or entry.upper == 'New York'):
            return 'NY'
        else:
            return entry
    elif math.isnan(entry):
        return ''
    else:
        return entry
data['state'] = data['state'].apply(cleaned_state)
data = data[data['state'] == 'NY']
state = {}
def create_state_set(entry):
    if (entry not in state):
        state[entry] = 1
    else:
        state[entry] += 1

data['state'].apply(create_state_set)
# print(state)


print('Spliting amenities')
amenities_set = set()

def collect_amenities(entry):
    entry_list = entry.replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace(" ", "_").split(',')
    for am in entry_list:
        if ('translation_missing' not in am and am != ''):
            amenities_set.add(am)

data['amenities'].apply(collect_amenities)
#print(amenities_set)


def generic_amenities(entry, amenity):
    entry_list = entry.replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace(" ", "_").split(',')
    for am in entry_list:
        if (am == amenity):
            return 1
    return 0

for amenity in amenities_set:
    data.insert(len(list(data)), amenity, 0)
    data[amenity] = data['amenities'].apply(lambda x: generic_amenities(x, amenity))

#print(data['amenities'][0])
#for v in  amenities_set:
#    print(v, " ", data[v][0])


#maybe drop the original column??
data = pd.DataFrame.drop(data, columns=['amenities', 'state'])

for col_name in ['property_type', 'bed_type',
                 'room_type', 'neighbourhood_group_cleansed', 'city',
                 'cancellation_policy', 'host_response_time', 'neighbourhood_cleansed']:
    parsed_cols = pd.get_dummies(data[col_name])
    data = data.drop(columns=[col_name])
    data = pd.concat([data, parsed_cols], axis = 1)

# Changing the host_since to number of days until 10 Nov 2018
def clean_host_since(entry):
    if (type(entry) != str and math.isnan(entry)):
        return -55
    return entry
data['host_since'] = data['host_since'].apply(clean_host_since)
data = data[data['host_since'] != -55]
dummy_date = dt.datetime(2018,11,10)
data.host_since = (dummy_date - pd.to_datetime(data.host_since))
data.host_since = data.host_since.apply(lambda x: float(x.days))



for col_name in ['review_scores_rating', 'review_scores_accuracy',
                 'review_scores_cleanliness', 'review_scores_checkin',
                 'review_scores_communication', 'review_scores_location',
                 'review_scores_value']:
    data[col_name] = data[col_name].apply(lambda x: 0 if np.isnan(x) else x)

data[col_name] = data[col_name].apply(lambda x: 0 if np.isnan(x) else x)

data = data.set_index('id').join(reviews.set_index('listing_id'))
def clean_comments(entry):
    if (type(entry) != str and math.isnan(entry)):
        return 0
    return entry
data['comments'] = data['comments'].apply(clean_comments)
data.to_csv('../Data/data_cleaned.csv')
