#!/usr/bin/env python3
"""
Comprehensive Test Runner for VisionSub Enhanced UI Testing

This script provides a comprehensive test runner for all enhanced UI components
with various test categories, reporting, and CI/CD integration capabilities.
"""

import sys
import os
import argparse
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "test_utils"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Comprehensive test runner for VisionSub UI testing"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load test configuration"""
        return {
            "test_categories": {
                "unit": {
                    "description": "Unit tests for individual components",
                    "files": ["test_enhanced_ui.py"],
                    "markers": ["unit"],
                    "timeout": 30
                },
                "integration": {
                    "description": "Integration tests for component interactions",
                    "files": ["test_integration.py"],
                    "markers": ["integration"],
                    "timeout": 60
                },
                "performance": {
                    "description": "Performance and benchmark tests",
                    "files": ["test_performance.py"],
                    "markers": ["performance"],
                    "timeout": 120
                },
                "security": {
                    "description": "Security and vulnerability tests",
                    "files": ["test_security.py"],
                    "markers": ["security"],
                    "timeout": 60
                },
                "accessibility": {
                    "description": "Accessibility and usability tests",
                    "files": ["test_enhanced_ui.py"],
                    "markers": ["accessibility"],
                    "timeout": 45
                },
                "e2e": {
                    "description": "End-to-end workflow tests",
                    "files": ["test_e2e.py"],
                    "markers": ["e2e"],
                    "timeout": 180
                },
                "smoke": {
                    "description": "Basic smoke tests",
                    "files": ["test_basic.py"],
                    "markers": ["smoke"],
                    "timeout": 30
                },
                "regression": {
                    "description": "Regression tests for known issues",
                    "files": ["test_enhanced_ui.py"],
                    "markers": ["regression"],
                    "timeout": 90
                },
                "comprehensive": {
                    "description": "Comprehensive test suite",
                    "files": ["test_enhanced_ui_comprehensive.py"],
                    "markers": [],
                    "timeout": 300
                }
            },
            "test_options": {
                "parallel": True,
                "parallel_workers": 4,
                "coverage": True,
                "coverage_threshold": 80,
                "html_report": True,
                "junit_report": True,
                "verbose": True,
                "fail_fast": False,
                "retry_count": 2,
                "screenshot_on_failure": True,
                "performance_report": True
            },
            "output_dirs": {
                "reports": "test_output/reports",
                "coverage": "test_output/coverage",
                "screenshots": "test_output/screenshots",
                "logs": "test_output/logs"
            }
        }
    
    def setup_environment(self):
        """Setup test environment"""
        logger.info("Setting up test environment...")
        
        # Create output directories
        for dir_path in self.config["output_dirs"].values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ.update({
            "VISIONUB_TEST_MODE": "1",
            "VISIONUB_LOG_LEVEL": "DEBUG",
            "VISIONUB_TEST_OUTPUT_DIR": "test_output",
            "QT_QPA_PLATFORM": "offscreen"  # For headless testing
        })
        
        logger.info("Test environment setup complete")
    
    def run_test_category(self, category: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests for a specific category"""
        logger.info(f"Running {category} tests...")
        
        category_config = self.config["test_categories"].get(category)
        if not category_config:
            logger.error(f"Unknown test category: {category}")
            return {"success": False, "error": f"Unknown category: {category}"}
        
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]
        
        # Add test files
        if category_config["files"]:
            cmd.extend(category_config["files"])
        
        # Add markers
        if category_config["markers"]:
            marker_expr = " or ".join(category_config["markers"])
            cmd.extend(["-m", marker_expr])
        
        # Add options
        cmd.extend(["-v", "--tb=short"])
        
        if options.get("parallel") and self.config["test_options"]["parallel"]:
            cmd.extend(["-n", str(self.config["test_options"]["parallel_workers"])])
        
        if options.get("coverage") and self.config["test_options"]["coverage"]:
            cmd.extend([
                "--cov=visionsub",
                "--cov-report=term-missing",
                f"--cov-report=html:{self.config['output_dirs']['coverage']}",
                f"--cov-fail-under={self.config['test_options']['coverage_threshold']}"
            ])
        
        if options.get("html_report") and self.config["test_options"]["html_report"]:
            cmd.extend([
                f"--html={self.config['output_dirs']['reports']}/{category}_report.html",
                "--self-contained-html"
            ])
        
        if options.get("junit_report") and self.config["test_options"]["junit_report"]:
            cmd.extend([
                f"--junitxml={self.config['output_dirs']['reports']}/{category}_junit.xml"
            ])
        
        if options.get("fail_fast") and self.config["test_options"]["fail_fast"]:
            cmd.append("--exitfirst")
        
        if options.get("screenshot_on_failure") and self.config["test_options"]["screenshot_on_failure"]:
            cmd.extend(["--screenshot-dir", self.config["output_dirs"]["screenshots"]])
        
        # Set timeout
        timeout = category_config.get("timeout", 60)
        
        # Run tests
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=project_root
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Parse results
            test_results = {
                "category": category,
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": " ".join(cmd)
            }
            
            logger.info(f"{category} tests completed in {execution_time:.2f}s")
            if result.returncode == 0:
                logger.info(f"✅ {category} tests passed")
            else:
                logger.error(f"❌ {category} tests failed")
                logger.error(f"Error output: {result.stderr}")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {category} tests timed out after {timeout}s")
            return {
                "category": category,
                "success": False,
                "error": f"Tests timed out after {timeout}s",
                "execution_time": timeout
            }
        except Exception as e:
            logger.error(f"❌ {category} tests failed with exception: {e}")
            return {
                "category": category,
                "success": False,
                "error": str(e),
                "execution_time": 0
            }
    
    def run_all_tests(self, categories: List[str] = None) -> Dict[str, Any]:
        """Run all tests or specified categories"""
        if categories is None:
            categories = list(self.config["test_categories"].keys())
        
        self.start_time = time.time()
        self.test_results = {}
        
        logger.info(f"🚀 Starting test run for categories: {', '.join(categories)}")
        
        for category in categories:
            result = self.run_test_category(category, self.config["test_options"])
            self.test_results[category] = result
            
            # Stop on failure if fail_fast is enabled
            if not result["success"] and self.config["test_options"]["fail_fast"]:
                logger.error(f"Stopping test run due to failure in {category}")
                break
        
        self.end_time = time.time()
        
        return self.test_results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive test summary report"""
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        # Calculate statistics
        total_categories = len(self.test_results)
        passed_categories = sum(1 for result in self.test_results.values() if result["success"])
        failed_categories = total_categories - passed_categories
        
        # Calculate total test counts (approximate)
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for result in self.test_results.values():
            if "stdout" in result:
                # Parse test counts from stdout (basic parsing)
                stdout = result["stdout"]
                if "passed" in stdout.lower():
                    # This is a very basic parsing - in practice you'd want more sophisticated parsing
                    passed_tests += 1
                if "failed" in stdout.lower():
                    failed_tests += 1
                total_tests += 1
        
        summary = {
            "test_run_info": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_time": total_time,
                "total_categories": total_categories,
                "passed_categories": passed_categories,
                "failed_categories": failed_categories,
                "success_rate": (passed_categories / total_categories * 100) if total_categories > 0 else 0
            },
            "test_statistics": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "category_results": self.test_results,
            "performance_metrics": self.calculate_performance_metrics(),
            "recommendations": self.generate_recommendations()
        }
        
        return summary
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from test results"""
        metrics = {
            "total_execution_time": 0,
            "average_execution_time": 0,
            "fastest_category": None,
            "slowest_category": None,
            "category_times": {}
        }
        
        total_time = 0
        fastest_time = float('inf')
        slowest_time = 0
        fastest_category = None
        slowest_category = None
        
        for category, result in self.test_results.items():
            if "execution_time" in result:
                execution_time = result["execution_time"]
                metrics["category_times"][category] = execution_time
                total_time += execution_time
                
                if execution_time < fastest_time:
                    fastest_time = execution_time
                    fastest_category = category
                
                if execution_time > slowest_time:
                    slowest_time = execution_time
                    slowest_category = category
        
        metrics["total_execution_time"] = total_time
        metrics["average_execution_time"] = total_time / len(self.test_results) if self.test_results else 0
        metrics["fastest_category"] = fastest_category
        metrics["slowest_category"] = slowest_category
        
        return metrics
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for failed categories
        failed_categories = [cat for cat, result in self.test_results.items() if not result["success"]]
        if failed_categories:
            recommendations.append(f"Failed test categories: {', '.join(failed_categories)}")
        
        # Check performance issues
        metrics = self.calculate_performance_metrics()
        if metrics["slowest_category"]:
            slow_time = metrics["category_times"].get(metrics["slowest_category"], 0)
            if slow_time > 60:  # More than 1 minute
                recommendations.append(f"Consider optimizing {metrics['slowest_category']} tests ({slow_time:.2f}s)")
        
        # Check coverage
        # This would need to be implemented by parsing coverage reports
        
        # Check for timeout issues
        timeout_categories = [
            cat for cat, result in self.test_results.items()
            if "error" in result and "timed out" in result.get("error", "")
        ]
        if timeout_categories:
            recommendations.append(f"Timeout issues in: {', '.join(timeout_categories)}")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save test report to file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"test_report_{timestamp}.json"
        
        report_path = Path(self.config["output_dirs"]["reports"]) / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Test report saved to: {report_path}")
        return report_path
    
    def print_summary(self, report: Dict[str, Any]):
        """Print test summary to console"""
        print("\n" + "="*60)
        print("🧪 VISIONUB ENHANCED UI TEST SUMMARY")
        print("="*60)
        
        # Test run info
        run_info = report["test_run_info"]
        print(f"📊 Total Test Time: {run_info['total_time']:.2f}s")
        print(f"📂 Categories: {run_info['passed_categories']}/{run_info['total_categories']} passed")
        print(f"✅ Success Rate: {run_info['success_rate']:.1f}%")
        
        # Test statistics
        stats = report["test_statistics"]
        print(f"🧮 Tests: {stats['passed_tests']}/{stats['total_tests']} passed")
        print(f"📈 Pass Rate: {stats['pass_rate']:.1f}%")
        
        # Performance metrics
        metrics = report["performance_metrics"]
        print(f"⏱️  Total Execution Time: {metrics['total_execution_time']:.2f}s")
        print(f"📊 Average Time per Category: {metrics['average_execution_time']:.2f}s")
        if metrics["fastest_category"]:
            print(f"🚀 Fastest Category: {metrics['fastest_category']} ({metrics['category_times'][metrics['fastest_category']]:.2f}s)")
        if metrics["slowest_category"]:
            print(f"🐌 Slowest Category: {metrics['slowest_category']} ({metrics['category_times'][metrics['slowest_category']]:.2f}s)")
        
        # Category results
        print("\n📋 Category Results:")
        for category, result in report["category_results"].items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            time_str = f"{result.get('execution_time', 0):.2f}s" if "execution_time" in result else "N/A"
            print(f"   {category}: {status} ({time_str})")
        
        # Recommendations
        if report["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
        
        print("="*60)
    
    def run_ci_tests(self) -> Dict[str, Any]:
        """Run tests suitable for CI/CD environment"""
        logger.info("Running CI/CD test suite...")
        
        # CI typically runs unit, integration, and security tests
        ci_categories = ["unit", "integration", "security", "smoke"]
        
        # Modify options for CI
        ci_options = self.config["test_options"].copy()
        ci_options.update({
            "parallel": True,
            "coverage": True,
            "html_report": True,
            "junit_report": True,
            "fail_fast": True,
            "screenshot_on_failure": True
        })
        
        results = {}
        for category in ci_categories:
            result = self.run_test_category(category, ci_options)
            results[category] = result
            
            # Stop on failure for CI
            if not result["success"]:
                logger.error(f"CI tests failed on {category}")
                break
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for VisionSub Enhanced UI Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                    # Run all tests
  %(prog)s --category unit          # Run unit tests only
  %(prog)s --category integration --category security  # Run specific categories
  %(prog)s --ci                     # Run CI/CD test suite
  %(prog)s --parallel --coverage    # Run with parallel execution and coverage
        """
    )
    
    parser.add_argument(
        "--all", action="store_true",
        help="Run all test categories"
    )
    
    parser.add_argument(
        "--category", action="append", dest="categories",
        help="Run specific test category (can be used multiple times)"
    )
    
    parser.add_argument(
        "--ci", action="store_true",
        help="Run CI/CD test suite"
    )
    
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Run tests sequentially"
    )
    
    parser.add_argument(
        "--coverage", action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--no-coverage", action="store_true",
        help="Skip coverage report"
    )
    
    parser.add_argument(
        "--html-report", action="store_true",
        help="Generate HTML report"
    )
    
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--output-dir", type=str,
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner()
    
    # Setup environment
    runner.setup_environment()
    
    # Configure options based on arguments
    options = runner.config["test_options"].copy()
    
    if args.parallel:
        options["parallel"] = True
    elif args.no_parallel:
        options["parallel"] = False
    
    if args.coverage:
        options["coverage"] = True
    elif args.no_coverage:
        options["coverage"] = False
    
    if args.html_report:
        options["html_report"] = True
    
    if args.fail_fast:
        options["fail_fast"] = True
    
    if args.output_dir:
        for dir_type in runner.config["output_dirs"]:
            runner.config["output_dirs"][dir_type] = os.path.join(args.output_dir, dir_type)
    
    # Run tests
    if args.ci:
        logger.info("🔄 Running CI/CD test suite...")
        results = runner.run_ci_tests()
    elif args.all:
        logger.info("🔄 Running all test categories...")
        results = runner.run_all_tests()
    elif args.categories:
        logger.info(f"🔄 Running test categories: {', '.join(args.categories)}...")
        results = {}
        for category in args.categories:
            result = runner.run_test_category(category, options)
            results[category] = result
    else:
        logger.info("🔄 Running default test suite (unit, integration, security)...")
        results = runner.run_all_tests(["unit", "integration", "security"])
    
    # Generate report
    report = runner.generate_summary_report()
    
    # Print summary
    runner.print_summary(report)
    
    # Save report
    report_path = runner.save_report(report)
    
    # Exit with appropriate code
    if all(result.get("success", False) for result in results.values()):
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()