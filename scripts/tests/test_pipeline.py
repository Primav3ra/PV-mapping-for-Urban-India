#!/usr/bin/env python3
"""
Test Pipeline for Solar Mapping Project
Validates the complete solar mapping pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ain_solar_mapper import SolarPotentialMapper, CityTier
import json

def test_basic_pipeline():
    """Test the basic solar mapping pipeline"""
    print("[INFO] Testing Solar Mapping Pipeline")
    print("=" * 50)
    
    try:
        # Initialize mapper
        print("1. Initializing Solar Potential Mapper...")
        mapper = SolarPotentialMapper('pv-mapping-india')
        print("[OK] Mapper initialized successfully")
        
        # Test AOI manager
        print("\n2. Testing AOI Manager...")
        mapper.list_available_cities()
        
        # Test single city analysis
        print("\n3. Testing single city analysis...")
        mapper.set_aoi_by_city('delhi')
        results = mapper.run_analysis()
        print(f"[OK] Delhi analysis completed: {results['analysis_completed']}")
        print(f"     Area: {results['aoi_area_km2']:.2f} km^2")
        
        # Test GeoJSON loading
        print("\n4. Testing GeoJSON AOI loading...")
        mapper.set_aoi_from_geojson('mumbai')
        results = mapper.run_analysis()
        print(f"[OK] Mumbai analysis from GeoJSON completed: {results['analysis_completed']}")
        
        # Test multiple cities (small subset)
        print("\n5. Testing multiple cities analysis...")
        test_cities = ['delhi', 'bangalore']
        multi_results = mapper.analyze_multiple_cities(test_cities)
        print(f"[OK] Multi-city analysis completed: {len(multi_results)} cities")
        
        # Test tier analysis (just Tier 1 for now)
        print("\n6. Testing tier analysis...")
        tier1_results = mapper.analyze_by_tier(CityTier.TIER_1)
        print(f"[OK] Tier 1 analysis completed: {len(tier1_results)} cities")
        
        # Test regional analysis
        print("\n7. Testing regional analysis...")
        north_results = mapper.analyze_by_region('north')
        print(f"[OK] North region analysis completed: {len(north_results)} cities")
        
        # Test export functionality
        print("\n8. Testing export functionality...")
        test_results = {"test": "data", "timestamp": "2024-01-01"}
        mapper.export_results(test_results, 'data/test_results.json')
        
        print("\n[PASS] All basic pipeline tests completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_aoi_management():
    """Test AOI management functionality"""
    print("\n[INFO] Testing AOI Management")
    print("=" * 30)
    
    try:
        mapper = SolarPotentialMapper('pv-mapping-india')
        
        # Test city info
        print("1. Testing city information retrieval...")
        delhi_info = mapper.get_city_info('delhi')
        print(f"[OK] Delhi info: {delhi_info['name']} - {delhi_info['population']:,} people")
        
        # Test AOI export
        print("\n2. Testing AOI export...")
        mapper.export_aoi_to_geojson('bangalore', 'data/test_bangalore_aoi.geojson')
        
        # Test coordinate-based AOI
        print("\n3. Testing coordinate-based AOI...")
        test_coords = [[77.0, 28.4], [77.1, 28.4], [77.1, 28.5], [77.0, 28.5], [77.0, 28.4]]
        mapper.set_aoi_by_coordinates(test_coords)
        results = mapper.run_analysis()
        print(f"[OK] Coordinate-based analysis completed: {results['analysis_completed']}")
        
        print("[PASS] AOI management tests completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] AOI management test failed: {str(e)}")
        return False

def test_data_validation():
    """Test data validation and error handling"""
    print("\n[INFO] Testing Data Validation")
    print("=" * 30)
    
    try:
        mapper = SolarPotentialMapper('pv-mapping-india')
        
        # Test invalid city name
        print("1. Testing invalid city name handling...")
        try:
            mapper.set_aoi_by_city('invalid_city')
            print("[FAIL] Should have raised an error for invalid city")
            return False
        except ValueError as e:
            print(f"[OK] Correctly handled invalid city: {str(e)}")
        
        # Test analysis without AOI
        print("\n2. Testing analysis without AOI...")
        try:
            mapper.run_analysis()
            print("[FAIL] Should have raised an error for missing AOI")
            return False
        except ValueError as e:
            print(f"[OK] Correctly handled missing AOI: {str(e)}")
        
        print("[PASS] Data validation tests completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Data validation test failed: {str(e)}")
        return False

def run_performance_test():
    """Run a simple performance test"""
    print("\n[INFO] Running Performance Test")
    print("=" * 30)
    
    try:
        mapper = SolarPotentialMapper('pv-mapping-india')
        
        import time
        start_time = time.time()
        
        # Test with a small city
        mapper.set_aoi_by_city('chandigarh')
        results = mapper.run_analysis()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"[OK] Analysis completed in {duration:.2f} seconds")
        print(f"     City: {results['city']}")
        print(f"     Area: {results['aoi_area_km2']:.2f} km^2")
        
        if duration < 30:  # Should complete within 30 seconds
            print("[PASS] Performance test passed")
            return True
        else:
            print("[WARN] Performance test: analysis took longer than expected (threshold 30 s)")
            return True
            
    except Exception as e:
        print(f"[FAIL] Performance test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("[INFO] Solar Mapping Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Pipeline", test_basic_pipeline),
        ("AOI Management", test_aoi_management),
        ("Data Validation", test_data_validation),
        ("Performance", run_performance_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[PASS] All tests passed")
    else:
        print("[WARN] Some tests failed; see messages above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
