#!/bin/bash

# VisionSub Test Runner Script
# Usage: ./run_tests.sh [test_type] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
COVERAGE=false
PARALLEL=false
VERBOSE=false
PERFORMANCE_ONLY=false
UNIT_ONLY=false
INTEGRATION_ONLY=false
REPORT_DIR="test_reports"

# Help function
show_help() {
    echo -e "${BLUE}VisionSub Test Runner${NC}"
    echo ""
    echo "Usage: $0 [test_type] [options]"
    echo ""
    echo "Test types:"
    echo "  all                    Run all tests (default)"
    echo "  unit                   Run unit tests only"
    echo "  integration            Run integration tests only"
    echo "  performance            Run performance tests only"
    echo "  smoke                  Run smoke tests only"
    echo "  e2e                    Run end-to-end tests only"
    echo ""
    echo "Options:"
    echo "  -c, --coverage          Generate coverage report"
    echo "  -p, --parallel          Run tests in parallel"
    echo "  -v, --verbose           Verbose output"
    echo "  --performance-only     Only run performance tests"
    echo "  --unit-only            Only run unit tests"
    echo "  --integration-only     Only run integration tests"
    echo "  --report-dir DIR       Report directory (default: test_reports)"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                      # Run all tests"
    echo "  $0 unit -c                             # Run unit tests with coverage"
    echo "  $0 integration -p                      # Run integration tests in parallel"
    echo "  $0 performance -v                      # Run performance tests with verbose output"
    echo "  $0 smoke -c -p                        # Run smoke tests with coverage and parallel"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        all|unit|integration|performance|smoke|e2e)
            TEST_TYPE=$1
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --performance-only)
            PERFORMANCE_ONLY=true
            shift
            ;;
        --unit-only)
            UNIT_ONLY=true
            shift
            ;;
        --integration-only)
            INTEGRATION_ONLY=true
            shift
            ;;
        --report-dir)
            REPORT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Set up environment
export VISIONSUB_ENVIRONMENT="test"
export VISIONSUB_LOG_LEVEL="DEBUG"
export VISIONSUB_TEST_MODE="true"

# Create report directory
mkdir -p "$REPORT_DIR"

# Function to run tests
run_tests() {
    local test_type=$1
    local extra_args=$2
    
    echo -e "${BLUE}Running $test_type tests...${NC}"
    
    case $test_type in
        "unit")
            markers="unit"
            ;;
        "integration")
            markers="integration"
            ;;
        "performance")
            markers="performance"
            ;;
        "smoke")
            markers="smoke"
            ;;
        "e2e")
            markers="e2e"
            ;;
        "all")
            markers="unit or integration or performance or smoke or e2e"
            ;;
    esac
    
    # Build pytest command
    cmd="pytest"
    
    # Add coverage if requested
    if $COVERAGE; then
        cmd="$cmd --cov=src/visionsub --cov-report=html:$REPORT_DIR/htmlcov --cov-report=xml:$REPORT_DIR/coverage.xml --cov-report=term-missing"
    fi
    
    # Add parallel if requested
    if $PARALLEL; then
        cmd="$cmd -n auto"
    fi
    
    # Add verbose if requested
    if $VERBOSE; then
        cmd="$cmd -v"
    fi
    
    # Add test selection
    if [[ "$test_type" != "all" ]]; then
        cmd="$cmd -m \"$markers\""
    fi
    
    # Add extra arguments
    cmd="$cmd $extra_args"
    
    # Run tests
    echo -e "${YELLOW}Running: $cmd${NC}"
    eval $cmd
    
    # Check exit code
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}$test_type tests passed!${NC}"
    else
        echo -e "${RED}$test_type tests failed!${NC}"
        return $exit_code
    fi
}

# Function to generate test report
generate_report() {
    echo -e "${BLUE}Generating test report...${NC}"
    
    # Create summary
    {
        echo "# VisionSub Test Report"
        echo ""
        echo "Generated on: $(date)"
        echo ""
        echo "## Test Results Summary"
        echo ""
        
        if [ -f "$REPORT_DIR/coverage.xml" ]; then
            echo "### Coverage Report"
            echo ""
            echo "Coverage XML report generated at: $REPORT_DIR/coverage.xml"
            echo "HTML coverage report available at: $REPORT_DIR/htmlcov/index.html"
            echo ""
        fi
        
        echo "### Test Categories"
        echo ""
        echo "- Unit Tests: Core component functionality"
        echo "- Integration Tests: Component interactions"
        echo "- Performance Tests: Benchmarking and performance metrics"
        echo "- Smoke Tests: Basic functionality verification"
        echo "- End-to-End Tests: Complete workflow testing"
        echo ""
        
        echo "### Test Environment"
        echo ""
        echo "- Python: $(python --version)"
        echo "- Platform: $(uname -s)"
        echo "- Architecture: $(uname -m)"
        echo ""
        
        echo "### Test Configuration"
        echo ""
        echo "- Coverage: $COVERAGE"
        echo "- Parallel: $PARALLEL"
        echo "- Verbose: $VERBOSE"
        echo ""
    } > "$REPORT_DIR/README.md"
    
    echo -e "${GREEN}Test report generated at: $REPORT_DIR/README.md${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}VisionSub Test Runner${NC}"
    echo "========================"
    echo ""
    echo "Test type: $TEST_TYPE"
    echo "Coverage: $COVERAGE"
    echo "Parallel: $PARALLEL"
    echo "Verbose: $VERBOSE"
    echo "Report directory: $REPORT_DIR"
    echo ""
    
    # Change to project root
    cd "$(dirname "$0")/.."
    
    # Install dependencies if needed
    if [ ! -d ".venv" ]; then
        echo -e "${YELLOW}Installing test dependencies...${NC}"
        poetry install --with dev,test
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Run specific test types
    exit_code=0
    
    if $UNIT_ONLY; then
        run_tests "unit" || exit_code=$?
    elif $INTEGRATION_ONLY; then
        run_tests "integration" || exit_code=$?
    elif $PERFORMANCE_ONLY; then
        run_tests "performance" || exit_code=$?
    else
        case $TEST_TYPE in
            "all")
                run_tests "unit" || exit_code=$?
                run_tests "integration" || exit_code=$?
                run_tests "performance" || exit_code=$?
                run_tests "smoke" || exit_code=$?
                run_tests "e2e" || exit_code=$?
                ;;
            *)
                run_tests "$TEST_TYPE" || exit_code=$?
                ;;
        esac
    fi
    
    # Generate report
    if $COVERAGE; then
        generate_report
    fi
    
    # Summary
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}Some tests failed!${NC}"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"