#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>

#include "interpolate.h"


TEST_CASE( "test_bisect_left", "[bisect_left]" ) {
    std::vector<float> list = {1., 2., 3., 4.};
    REQUIRE( bisect_left(list.begin(), list.end(), 0.5) == 0 );
    REQUIRE( bisect_left(list.begin(), list.end(), 1.5) == 1 );
    REQUIRE( bisect_left(list.begin(), list.end(), 2.1) == 2 );
    REQUIRE( bisect_left(list.begin(), list.end(), 3.99) == 3 );
    REQUIRE( bisect_left(list.begin(), list.end(), 4.2) == 4 );
}

TEST_CASE( "test_find_left", "[find_left]" ) {
    std::vector<float> list = {1., 2., 3., 4.};
    REQUIRE( find_left(list.begin(), list.end(), 0.5) == 0 );
    REQUIRE( find_left(list.begin(), list.end(), 1.5) == 1 );
    REQUIRE( find_left(list.begin(), list.end(), 2.1) == 2 );
    REQUIRE( find_left(list.begin(), list.end(), 3.99) == 3 );
    REQUIRE( find_left(list.begin(), list.end(), 4.2) == 4 );
}
