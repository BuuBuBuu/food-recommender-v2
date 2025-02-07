# utils.py
import json
import os
import logging
from datetime import datetime
import googlemaps
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'food_recommender_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

def analyze_query_intent(query: str):
    """
    Analyze if the query is food-related and extract location information.
    Now includes friendly responses for non-food queries.
    """
    system_msg = """
You are a friendly expert food recommendation assistant. Your task is to:
1. Analyze queries to understand user intent
2. For food-related queries:
   - Identify specific dishes
   - Extract and validate location information
   - Determine search scope based on location specificity
3. For non-food queries:
   - Respond warmly while guiding user towards food recommendations

Response format (JSON):
{
    "is_food_query": boolean,
    "message": string,
    "parsed_data": {
        "food_item": string,
        "location_data": {
            "provided": boolean,
            "type": "country" | "city" | "district" | "area",
            "name": string,
            "country": string,
            "subdivision": string (optional)
        },
        "numeric_price_level": integer (0-4) or null,
        "search_terms": array of strings
    }
}

Examples:
1. Food query with location:
"Best chicken rice at Tanjong Pagar Singapore"
{
    "is_food_query": true,
    "message": null,
    "parsed_data": {
        "food_item": "chicken rice",
        "location_data": {
            "provided": true,
            "type": "district",
            "name": "Tanjong Pagar",
            "country": "Singapore"
        },
        "numeric_price_level": null,
        "search_terms": ["chicken rice", "Tanjong Pagar", "Singapore"]
    }
}

2. Food query without location:
"Sushi"
{
    "is_food_query": true,
    "message": "I'd love to help you find great sushi! Could you let me know where you'd like to eat? You can specify a country, city, or specific area.",
    "parsed_data": {
        "food_item": "sushi",
        "location_data": {
            "provided": false
        },
        "numeric_price_level": null,
        "search_terms": ["sushi"]
    }
}

3. Non-food query:
"Hello how are you?"
{
    "is_food_query": false,
    "message": "Hello! I'm here to help you discover great food places. Are you looking for any particular cuisine today? Just let me know what you'd like to eat and where!",
    "parsed_data": null
}

4. "Italian restaurants in London"
{
    "is_food_query": true,
    "message": null,
    "parsed_data": {
        "food_item": "Italian food",
        "location_data": {
            "provided": true,
            "type": "city",
            "name": "London",
            "country": "United Kingdom"
        },
        "numeric_price_level": null,
        "search_terms": ["Italian food", "London", "United Kingdom"]
    }
}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Analyze this query: '{query}'"}
            ],
            temperature=0.7,
            max_tokens=300
        )

        result = json.loads(response.choices[0].message.content)

        if not result["is_food_query"]:
            raise ValueError(result["message"])

        if result["is_food_query"] and not result["parsed_data"]["location_data"]["provided"]:
            raise ValueError(result["message"] or "Please specify a location for your search. For example:\n- 'Best pizza in New York'\n- 'Sushi restaurants in Tokyo, Japan'\n- 'Local food in Singapore'")

        logger.info(f"Query analysis: {json.dumps(result['parsed_data'], indent=2)}")
        return result

    except ValueError as e:
        raise ValueError(str(e))
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise

def geocode_location(location):
    """Get coordinates for a location."""
    logger.info(f"Geocoding: {location}")

    try:
        geocode_result = gmaps.geocode(location)
        if geocode_result:
            loc = geocode_result[0]["geometry"]["location"]
            logger.info(f"Coordinates found: {loc}")
            return loc["lat"], loc["lng"]
        logger.warning(f"Location not found: {location}")

    except Exception as e:
        logger.error(f"Geocoding error: {e}")
    return None, None

def search_restaurants_by_location(food_item, location_data):
    """
    Search for restaurants based on location granularity.
    Adapts search radius and strategy based on location type.
    """
    logger.info(f"Searching for {food_item} in {location_data['name']}")

    # Construct location string based on available data
    location_parts = [location_data['name']]
    if location_data.get('subdivision'):
        location_parts.append(location_data['subdivision'])
    location_parts.append(location_data['country'])
    location_string = ", ".join(location_parts)

    # Get coordinates
    lat, lng = geocode_location(location_string)
    if not (lat and lng):
        raise ValueError(f"Could not find location: {location_string}")

    # Adjust search parameters based on location type
    search_params = {
        'country': {'radius': 50000, 'max_results': 30},
        'city': {'radius': 15000, 'max_results': 25},
        'district': {'radius': 5000, 'max_results': 20},
        'area': {'radius': 2000, 'max_results': 15}
    }

    params = search_params.get(location_data['type'],
                             {'radius': 5000, 'max_results': 20})

    try:
        places_result = gmaps.places_nearby(
            location=(lat, lng),
            radius=params['radius'],
            keyword=food_item,
            type="restaurant"
        )

        if not places_result.get('results'):
            return []

        results = places_result.get('results', [])[:params['max_results']]
        return process_place_details(results, food_item, location_data)

    except Exception as e:
        logger.error(f"Error in restaurant search: {e}")
        return []

def get_place_details(place_id):
    """
    Get detailed information for a specific place.
    """
    try:
        place_details_resp = gmaps.place(
            place_id,
            fields=[
                "name", "rating", "user_ratings_total", "formatted_address",
                "price_level", "opening_hours", "vicinity", "type",
                "international_phone_number", "website", "photo",
                "geometry", "reviews"
            ]
        )

        if "result" in place_details_resp:
            return place_details_resp["result"]
    except Exception as e:
        logger.error(f"Error getting place details: {e}")
    return None

def process_reviews(reviews, keyword):
    """
    Process reviews and calculate keyword relevance.
    """
    processed_reviews = []
    keyword_mentions = 0

    for review in reviews:
        processed_review = {
            "author_name": review.get("author_name", ""),
            "rating": review.get("rating", 0),
            "text": review.get("text", ""),
            "time": review.get("time", ""),
            "relative_time_description": review.get("relative_time_description", "")
        }

        if keyword.lower() in review.get("text", "").lower():
            keyword_mentions += 1

        processed_reviews.append(processed_review)

    return {
        "processed_reviews": processed_reviews,
        "review_count": len(processed_reviews),
        "keyword_relevance": keyword_mentions / len(processed_reviews) if processed_reviews else 0
    }

def process_place_details(places, food_item, location_data):
    """
    Process and enrich place details with additional context.
    """
    detailed_results = []

    for place in places:
        try:
            place_details = get_place_details(place['place_id'])
            if place_details:
                enriched_place = place.copy()
                enriched_place.update(place_details)

                # Add location context
                enriched_place['location_context'] = {
                    'search_area': location_data['name'],
                    'location_type': location_data['type'],
                    'country': location_data['country']
                }

                # Process reviews and calculate relevance
                if 'reviews' in place_details:
                    enriched_place.update(
                        process_reviews(place_details['reviews'], food_item)
                    )

                detailed_results.append(enriched_place)
                logger.info(f"Processed: {enriched_place.get('name')} (rating: {enriched_place.get('rating')})")

        except Exception as e:
            logger.error(f"Error processing place {place.get('name')}: {e}")
            continue

    logger.info(f"Successfully processed {len(detailed_results)} restaurants")
    return detailed_results
