import pandas as pd
import numpy as np

# to calculate yield for a given crop of a state, we actually need yield/area_
def yield_for_crop(df, crop_name, state, year):
    #filtering the data for therequired crop and state 
    # we can take the mean for the years given as the yield
    crop_df = df[df['crop'].str.lower().str.contains(crop_name.lower(), na=False) & 
                 df['state_name'].str.lower().str.contains(state.lower(), na=False) & 
                 df['crop_year'].astype(str).str.contains(year.lower(), na=False)]
    print(crop_df)

    #aggregate total area and total production
    total_area = crop_df['area_'].sum()
    total_production_tonnes = crop_df['yield'].sum()

    #covert tonnes to quintal
    total_production_quintal = total_production_tonnes * 10

    #yield = production / area
    yield_q_he = total_production_quintal / total_area
    return yield_q_he
    
            

if __name__ == "__main__":
    df = pd.read_csv("data/yield_data.csv")
    y = yield_for_crop(df, "rice", state="West Bengal", year="2010")
    print("Yield of rice in West Bengal in 2010 (quintal/hectare): ", y)
