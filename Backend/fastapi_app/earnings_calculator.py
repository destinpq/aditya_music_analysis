"""
Earnings calculator for YouTube videos based on view count and CPM rates.
Includes country-specific CPM rates and geographic distribution modeling.
"""

class YouTubeEarningsCalculator:
    """Calculate estimated earnings for YouTube videos"""
    
    def __init__(self):
        # Default CPM rates by country (in USD)
        self.cpm_rates = {
            # North America
            "US": {"min": 3.00, "avg": 4.50, "max": 7.00},
            "CA": {"min": 2.00, "avg": 3.25, "max": 5.00},
            "MX": {"min": 0.80, "avg": 1.20, "max": 2.00},
            
            # Europe
            "UK": {"min": 2.00, "avg": 3.00, "max": 5.00},
            "DE": {"min": 1.80, "avg": 2.50, "max": 4.50},
            "FR": {"min": 1.50, "avg": 2.25, "max": 4.00},
            "IT": {"min": 1.20, "avg": 1.80, "max": 3.50},
            "ES": {"min": 1.20, "avg": 1.75, "max": 3.50},
            "NL": {"min": 1.50, "avg": 2.25, "max": 3.75},
            "SE": {"min": 1.80, "avg": 2.50, "max": 4.00},
            
            # Asia-Pacific
            "IN": {"min": 0.30, "avg": 0.50, "max": 1.00},  # India (specifically included as requested)
            "JP": {"min": 1.50, "avg": 2.00, "max": 3.50},
            "KR": {"min": 1.20, "avg": 1.75, "max": 3.00},
            "AU": {"min": 1.80, "avg": 2.50, "max": 4.50},
            "SG": {"min": 1.00, "avg": 1.50, "max": 2.50},
            
            # Latin America
            "BR": {"min": 0.60, "avg": 1.00, "max": 1.80},
            "AR": {"min": 0.50, "avg": 0.80, "max": 1.50},
            
            # Middle East
            "AE": {"min": 1.00, "avg": 1.75, "max": 3.00},
            "SA": {"min": 0.90, "avg": 1.50, "max": 2.75},
            
            # Default for other countries
            "Global": {"min": 2.00, "avg": 3.00, "max": 5.00}
        }
        
        # Default geographic distribution of viewers (approximations)
        self.default_geography = {
            "US": 0.30,
            "UK": 0.06,
            "CA": 0.04,
            "IN": 0.12,
            "DE": 0.05,
            "FR": 0.04,
            "BR": 0.04,
            "MX": 0.03,
            "AU": 0.03,
            "JP": 0.03,
            "Other": 0.26  # Remaining distributed across other countries
        }
        
        # Default monetization parameters
        self.default_monetization_rate = 0.85  # % of views that are monetized
        self.default_ad_impression_rate = 0.70  # % of monetized views that see ads
    
    def calculate_earnings(self, view_count, custom_cpm=None, country=None, 
                          geography=None, monetization_rate=None, 
                          ad_impression_rate=None):
        """
        Calculate estimated YouTube earnings
        
        Parameters:
        -----------
        view_count : int
            Number of video views
        custom_cpm : float, optional
            Custom CPM rate to use (overrides country-specific rates)
        country : str, optional
            Specific country code to calculate for (ignores geography distribution)
        geography : dict, optional
            Custom geographic distribution of viewers
        monetization_rate : float, optional
            Percentage of views that can be monetized (0-1)
        ad_impression_rate : float, optional
            Percentage of monetized views that see ads (0-1)
            
        Returns:
        --------
        dict
            Earnings calculation results
        """
        # Input validation
        if not isinstance(view_count, (int, float)):
            try:
                view_count = int(float(view_count))
            except (ValueError, TypeError):
                return {"error": "Invalid view count provided"}
                
        view_count = max(0, int(view_count))
        
        # Set default parameters if not provided
        monetization_rate = monetization_rate or self.default_monetization_rate
        ad_impression_rate = ad_impression_rate or self.default_ad_impression_rate
        geography = geography or self.default_geography
        
        # Calculate monetized views
        monetized_views = view_count * monetization_rate
        ad_views = monetized_views * ad_impression_rate
        
        # Case 1: Using a custom CPM rate
        if custom_cpm is not None:
            try:
                custom_cpm = float(custom_cpm)
            except (ValueError, TypeError):
                return {"error": "Invalid CPM value provided"}
                
            earnings = (ad_views / 1000) * custom_cpm
            
            return {
                "view_count": view_count,
                "monetized_views": int(monetized_views),
                "ad_impression_views": int(ad_views),
                "monetization_rate": monetization_rate,
                "ad_impression_rate": ad_impression_rate,
                "cpm": custom_cpm,
                "estimated_earnings": round(earnings, 2)
            }
        
        # Case 2: Calculate for a specific country
        if country:
            country_code = country.upper()
            if country_code not in self.cpm_rates:
                country_code = "Global"
                
            cpm_data = self.cpm_rates[country_code]
            min_earnings = (ad_views / 1000) * cpm_data["min"]
            avg_earnings = (ad_views / 1000) * cpm_data["avg"] 
            max_earnings = (ad_views / 1000) * cpm_data["max"]
            
            return {
                "view_count": view_count,
                "monetized_views": int(monetized_views),
                "ad_impression_views": int(ad_views),
                "monetization_rate": monetization_rate,
                "ad_impression_rate": ad_impression_rate,
                "country": country_code,
                "cpm_range": cpm_data,
                "estimated_earnings": {
                    "min": round(min_earnings, 2),
                    "average": round(avg_earnings, 2),
                    "max": round(max_earnings, 2)
                }
            }
        
        # Case 3: Calculate based on geographic distribution
        earnings_by_country = {}
        total_min_earnings = 0
        total_avg_earnings = 0
        total_max_earnings = 0
        
        # Normalize geography distribution
        total_distribution = sum(geography.values())
        if abs(total_distribution - 1.0) > 0.01:  # Allow small rounding errors
            normalized_geography = {k: v/total_distribution for k, v in geography.items()}
        else:
            normalized_geography = geography
        
        # Calculate for each country in the distribution
        for country, percentage in normalized_geography.items():
            country_code = country.upper()
            if country_code not in self.cpm_rates:
                if country_code == "OTHER" or country_code == "OTHERS":
                    country_code = "Global"
                else:
                    country_code = "Global"
            
            country_views = ad_views * percentage
            country_cpm = self.cpm_rates[country_code]
            
            min_earnings = (country_views / 1000) * country_cpm["min"]
            avg_earnings = (country_views / 1000) * country_cpm["avg"]
            max_earnings = (country_views / 1000) * country_cpm["max"]
            
            earnings_by_country[country_code] = {
                "views": int(country_views),
                "distribution": percentage,
                "cpm": country_cpm,
                "earnings": {
                    "min": round(min_earnings, 2),
                    "average": round(avg_earnings, 2),
                    "max": round(max_earnings, 2)
                }
            }
            
            total_min_earnings += min_earnings
            total_avg_earnings += avg_earnings
            total_max_earnings += max_earnings
        
        return {
            "view_count": view_count,
            "monetized_views": int(monetized_views),
            "ad_impression_views": int(ad_views),
            "monetization_rate": monetization_rate,
            "ad_impression_rate": ad_impression_rate,
            "geography_distribution": normalized_geography,
            "estimated_earnings": {
                "min": round(total_min_earnings, 2),
                "average": round(total_avg_earnings, 2),
                "max": round(total_max_earnings, 2)
            },
            "earnings_by_country": earnings_by_country
        }
    
    def get_cpm_rates(self, country=None):
        """Get CPM rates for all countries or a specific country"""
        if country:
            country_code = country.upper()
            if country_code in self.cpm_rates:
                return {country_code: self.cpm_rates[country_code]}
            else:
                return {"Global": self.cpm_rates["Global"]}
        else:
            return self.cpm_rates
    
    def update_cpm_rate(self, country, min_rate, avg_rate, max_rate):
        """Update CPM rate for a specific country"""
        country_code = country.upper()
        self.cpm_rates[country_code] = {
            "min": float(min_rate),
            "avg": float(avg_rate),
            "max": float(max_rate)
        }
        return self.cpm_rates[country_code]

# Create a singleton instance
earnings_calculator = YouTubeEarningsCalculator() 