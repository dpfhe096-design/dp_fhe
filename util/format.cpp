#include <string>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

#include "format.h"

std::string getCurrentTime() {
    // Get current time as time_point
    auto now = std::chrono::system_clock::now();

    // Convert to time_t (calendar time)
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    // Convert to local time
    std::tm tm = *std::localtime(&t);

    // Format time into string
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}
