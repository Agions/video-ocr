#!/usr/bin/env python3
"""
Specific Test Category Runner for VisionSub Enhanced UI Testing

This script provides targeted test execution for specific test categories
with detailed reporting and analysis capabilities.
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


class CategoryTestRunner:
    """Runner for specific test categories with detailed analysis"""
    
    def __init__(self):
        self.category_configs = {
            "unit": {
                "description": "Unit tests for individual components",
                "files": ["test_enhanced_ui.py"],
                "markers": ["unit"],
                "timeout": 30,
                "priority": "high",
                "critical": True
            },
            "integration": {
                "description": "Integration tests for component interactions",
                "files": ["test_integration.py"],
                "markers": ["integration"],
                "timeout": 60,
                "priority": "high",
                "critical": True
            },
            "performance": {
                "description": "Performance and benchmark tests",
                "files": ["test_performance.py"],
                "markers": ["performance"],
                "timeout": 120,
                "priority": "medium",
                "critical": False
            },
            "security": {
                "description": "Security and vulnerability tests",
                "files": ["test_security.py"],
                "markers": ["security"],
                "timeout": 60,
                "priority": "high",
                "critical": True
            },
            "accessibility": {
                "description": "Accessibility and usability tests",
                "files": ["test_enhanced_ui.py"],
                "markers": ["accessibility"],
                "timeout": 45,
                "priority": "medium",
                "critical": False
            },
            "e2e": {
                "description": "End-to-end workflow tests",
                "files": ["test_e2e.py"],
                "markers": ["e2e"],
                "timeout": 180,
                "priority": "medium",
                "critical": False
            },
            "smoke": {
                "description": "Basic smoke tests",
                "files": ["test_basic.py"],
                "markers": ["smoke"],
                "timeout": 30,
                "priority": "high",
                "critical": True
            },
            "regression": {
                "description": "Regression tests for known issues",
                "files": ["test_enhanced_ui.py"],
                "markers": ["regression"],
                "timeout": 90,
                "priority": "medium",
                "critical": False
            },
            "comprehensive": {
                "description": "Comprehensive test suite",
                "files": ["test_enhanced_ui_comprehensive.py"],
                "markers": [],
                "timeout": 300,
                "priority": "low",
                "critical": False
            }
        }
        
        self.test_results = {}
        self.analysis_results = {}
        
    def setup_environment(self):
        """Setup test environment"""
        logger.info("Setting up test environment...")
        
        # Create output directories
        output_dirs = [
            "test_output/reports",
            "test_output/coverage",
            "test_output/screenshots",
            "test_output/logs"
        ]
        
        for dir_path in output_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ.update({
            "VISIONUB_TEST_MODE": "1",
            "VISIONUB_LOG_LEVEL": "DEBUG",
            "VISIONUB_TEST_OUTPUT_DIR": "test_output",
            "QT_QPA_PLATFORM": "offscreen"
        })
        
        logger.info("Test environment setup complete")
    
    def run_category_tests(self, category: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests for a specific category"""
        logger.info(f"Running {category} tests...")
        
        if category not in self.category_configs:
            logger.error(f"Unknown test category: {category}")
            return {"success": False, "error": f"Unknown category: {category}"}
        
        config = self.category_configs[category]
        
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]
        
        # Add test files
        if config["files"]:
            cmd.extend(config["files"])
        
        # Add markers
        if config["markers"]:
            marker_expr = " or ".join(config["markers"])
            cmd.extend(["-m", marker_expr])
        
        # Add options
        cmd.extend(["-v", "--tb=short"])
        
        # Add coverage
        if options.get("coverage", True):
            cmd.extend([
                "--cov=visionsub",
                "--cov-report=term-missing",
                "--cov-report=html:test_output/coverage",
                "--cov-fail-under=80"
            ])
        
        # Add HTML report
        if options.get("html_report", True):
            cmd.extend([
                f"--html=test_output/reports/{category}_report.html",
                "--self-contained-html"
            ])
        
        # Add parallel execution
        if options.get("parallel", False):
            cmd.extend(["-n", "4"])
        
        # Add timeout
        timeout = config.get("timeout", 60)
        
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
                "command": " ".join(cmd),
                "config": config
            }
            
            # Parse test statistics
            test_results.update(self.parse_test_statistics(result.stdout))
            
            logger.info(f"{category} tests completed in {execution_time:.2f}s")
            if result.returncode == 0:
                logger.info(f"✅ {category} tests passed")
            else:
                logger.error(f"❌ {category} tests failed")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {category} tests timed out after {timeout}s")
            return {
                "category": category,
                "success": False,
                "error": f"Tests timed out after {timeout}s",
                "execution_time": timeout,
                "config": config
            }
        except Exception as e:
            logger.error(f"❌ {category} tests failed with exception: {e}")
            return {
                "category": category,
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "config": config
            }
    
    def parse_test_statistics(self, stdout: str) -> Dict[str, Any]:
        """Parse test statistics from pytest output"""
        stats = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "error_tests": 0,
            "test_duration": 0.0,
            "coverage_percentage": 0.0
        }
        
        lines = stdout.split('\n')
        for line in lines:
            # Parse test counts
            if "passed" in line.lower() and ("failed" in line.lower() or "error" in line.lower()):
                # Example: "12 passed, 2 failed, 1 skipped in 5.23s"
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if "passed" in part:
                        try:
                            stats["passed_tests"] = int(part.split()[0])
                        except:
                            pass
                    elif "failed" in part:
                        try:
                            stats["failed_tests"] = int(part.split()[0])
                        except:
                            pass
                    elif "skipped" in part:
                        try:
                            stats["skipped_tests"] = int(part.split()[0])
                        except:
                            pass
                    elif "error" in part:
                        try:
                            stats["error_tests"] = int(part.split()[0])
                        except:
                            pass
            
            # Parse duration
            if "in " in line and "s" in line and ("passed" in line or "failed" in line):
                try:
                    duration_str = line.split("in ")[-1].replace("s", "")
                    stats["test_duration"] = float(duration_str)
                except:
                    pass
            
            # Parse coverage
            if "coverage" in line.lower() and "%" in line:
                try:
                    coverage_str = line.split("%")[0].split()[-1]
                    stats["coverage_percentage"] = float(coverage_str)
                except:
                    pass
        
        stats["total_tests"] = stats["passed_tests"] + stats["failed_tests"] + stats["skipped_tests"] + stats["error_tests"]
        
        return stats
    
    def run_multiple_categories(self, categories: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        """Run multiple test categories"""
        logger.info(f"Running test categories: {', '.join(categories)}")
        
        results = {}
        start_time = time.time()
        
        for category in categories:
            result = self.run_category_tests(category, options)
            results[category] = result
            
            # Stop on failure if critical test fails
            if not result["success"] and result["config"].get("critical", False):
                logger.error(f"Stopping test run due to critical failure in {category}")
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        analysis = self.analyze_results(results, total_time)
        
        return {
            "results": results,
            "analysis": analysis,
            "total_time": total_time
        }
    
    def analyze_results(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Analyze test results and provide insights"""
        analysis = {
            "summary": {
                "total_categories": len(results),
                "passed_categories": sum(1 for r in results.values() if r["success"]),
                "failed_categories": sum(1 for r in results.values() if not r["success"]),
                "total_time": total_time,
                "success_rate": 0.0
            },
            "statistics": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "error_tests": 0,
                "overall_coverage": 0.0
            },
            "performance": {
                "fastest_category": None,
                "slowest_category": None,
                "average_time": 0.0,
                "category_times": {}
            },
            "critical_failures": [],
            "recommendations": []
        }
        
        # Calculate summary
        analysis["summary"]["success_rate"] = (
            analysis["summary"]["passed_categories"] / analysis["summary"]["total_categories"] * 100
            if analysis["summary"]["total_categories"] > 0 else 0
        )
        
        # Calculate statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        coverage_sum = 0
        coverage_count = 0
        
        fastest_time = float('inf')
        slowest_time = 0
        fastest_category = None
        slowest_category = None
        
        for category, result in results.items():
            # Test statistics
            if "total_tests" in result:
                total_tests += result["total_tests"]
                passed_tests += result["passed_tests"]
                failed_tests += result["failed_tests"]
            
            # Coverage
            if "coverage_percentage" in result and result["coverage_percentage"] > 0:
                coverage_sum += result["coverage_percentage"]
                coverage_count += 1
            
            # Performance
            if "execution_time" in result:
                exec_time = result["execution_time"]
                analysis["performance"]["category_times"][category] = exec_time
                
                if exec_time < fastest_time:
                    fastest_time = exec_time
                    fastest_category = category
                
                if exec_time > slowest_time:
                    slowest_time = exec_time
                    slowest_category = category
            
            # Critical failures
            if not result["success"] and result["config"].get("critical", False):
                analysis["critical_failures"].append(category)
        
        # Update statistics
        analysis["statistics"]["total_tests"] = total_tests
        analysis["statistics"]["passed_tests"] = passed_tests
        analysis["statistics"]["failed_tests"] = failed_tests
        analysis["statistics"]["overall_coverage"] = (
            coverage_sum / coverage_count if coverage_count > 0 else 0
        )
        
        # Update performance
        analysis["performance"]["fastest_category"] = fastest_category
        analysis["performance"]["slowest_category"] = slowest_category
        analysis["performance"]["average_time"] = (
            total_time / len(results) if results else 0
        )
        
        # Generate recommendations
        analysis["recommendations"] = self.generate_recommendations(results, analysis)
        
        return analysis
    
    def generate_recommendations(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Critical failures
        if analysis["critical_failures"]:
            recommendations.append(
                f"🚨 Critical test failures in: {', '.join(analysis['critical_failures'])}"
            )
        
        # Low success rate
        if analysis["summary"]["success_rate"] < 80:
            recommendations.append(
                f"⚠️ Low success rate: {analysis['summary']['success_rate']:.1f}%"
            )
        
        # Low coverage
        if analysis["statistics"]["overall_coverage"] < 70:
            recommendations.append(
                f"📊 Low test coverage: {analysis['statistics']['overall_coverage']:.1f}%"
            )
        
        # Performance issues
        if analysis["performance"]["slowest_category"]:
            slow_time = analysis["performance"]["category_times"][analysis["performance"]["slowest_category"]]
            if slow_time > 120:  # More than 2 minutes
                recommendations.append(
                    f"🐌 Performance issue in {analysis['performance']['slowest_category']} ({slow_time:.2f}s)"
                )
        
        # High failure rate
        if analysis["statistics"]["total_tests"] > 0:
            failure_rate = analysis["statistics"]["failed_tests"] / analysis["statistics"]["total_tests"] * 100
            if failure_rate > 10:
                recommendations.append(
                    f"❌ High test failure rate: {failure_rate:.1f}%"
                )
        
        # Missing test categories
        missing_categories = set(self.category_configs.keys()) - set(results.keys())
        if missing_categories:
            recommendations.append(
                f"📋 Missing test categories: {', '.join(missing_categories)}"
            )
        
        return recommendations
    
    def print_category_report(self, category: str, result: Dict[str, Any]):
        """Print detailed report for a single category"""
        config = result["config"]
        
        print(f"\n{'='*60}")
        print(f"📊 {category.upper()} TEST REPORT")
        print(f"{'='*60}")
        print(f"Description: {config['description']}")
        print(f"Priority: {config['priority']}")
        print(f"Critical: {'Yes' if config['critical'] else 'No'}")
        print(f"Status: {'✅ PASS' if result['success'] else '❌ FAIL'}")
        print(f"Execution Time: {result.get('execution_time', 0):.2f}s")
        
        if "total_tests" in result:
            print(f"Tests: {result['passed_tests']}/{result['total_tests']} passed")
            if result['failed_tests'] > 0:
                print(f"Failed: {result['failed_tests']}")
            if result['skipped_tests'] > 0:
                print(f"Skipped: {result['skipped_tests']}")
            if result['error_tests'] > 0:
                print(f"Errors: {result['error_tests']}")
        
        if "coverage_percentage" in result and result['coverage_percentage'] > 0:
            print(f"Coverage: {result['coverage_percentage']:.1f}%")
        
        if not result["success"] and result.get("stderr"):
            print(f"\n❌ Error Output:")
            print(result["stderr"])
        
        print(f"{'='*60}")
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """Print analysis report"""
        print(f"\n{'='*60}")
        print("📈 TEST ANALYSIS REPORT")
        print(f"{'='*60}")
        
        # Summary
        summary = analysis["summary"]
        print(f"📊 Summary:")
        print(f"   Categories: {summary['passed_categories']}/{summary['total_categories']} passed")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Total Time: {summary['total_time']:.2f}s")
        
        # Statistics
        stats = analysis["statistics"]
        print(f"\n📈 Statistics:")
        print(f"   Total Tests: {stats['total_tests']}")
        print(f"   Passed: {stats['passed_tests']}")
        print(f"   Failed: {stats['failed_tests']}")
        print(f"   Coverage: {stats['overall_coverage']:.1f}%")
        
        # Performance
        perf = analysis["performance"]
        print(f"\n⏱️ Performance:")
        print(f"   Average Time: {perf['average_time']:.2f}s")
        if perf['fastest_category']:
            print(f"   Fastest: {perf['fastest_category']} ({perf['category_times'][perf['fastest_category']]:.2f}s)")
        if perf['slowest_category']:
            print(f"   Slowest: {perf['slowest_category']} ({perf['category_times'][perf['slowest_category']]:.2f}s)")
        
        # Recommendations
        if analysis["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"   • {rec}")
        
        print(f"{'='*60}")
    
    def save_report(self, results: Dict[str, Any], analysis: Dict[str, Any], filename: str = None):
        """Save detailed report to file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"category_test_report_{timestamp}.json"
        
        report_path = Path("test_output/reports") / filename
        
        report = {
            "timestamp": time.time(),
            "results": results,
            "analysis": analysis
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {report_path}")
        return report_path


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run specific test categories for VisionSub Enhanced UI Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --category unit                    # Run unit tests
  %(prog)s --category unit --category security  # Run multiple categories
  %(prog)s --all                              # Run all categories
  %(prog)s --critical                         # Run only critical categories
  %(prog)s --high-priority                    # Run high priority categories
        """
    )
    
    parser.add_argument(
        "--category", action="append", dest="categories",
        help="Run specific test category (can be used multiple times)"
    )
    
    parser.add_argument(
        "--all", action="store_true",
        help="Run all test categories"
    )
    
    parser.add_argument(
        "--critical", action="store_true",
        help="Run only critical test categories"
    )
    
    parser.add_argument(
        "--high-priority", action="store_true",
        help="Run only high priority test categories"
    )
    
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--no-coverage", action="store_true",
        help="Skip coverage report"
    )
    
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip HTML report generation"
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--save-report", action="store_true",
        help="Save detailed report to file"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = CategoryTestRunner()
    
    # Setup environment
    runner.setup_environment()
    
    # Determine categories to run
    if args.all:
        categories = list(runner.category_configs.keys())
    elif args.critical:
        categories = [
            cat for cat, config in runner.category_configs.items()
            if config.get("critical", False)
        ]
    elif args.high_priority:
        categories = [
            cat for cat, config in runner.category_configs.items()
            if config.get("priority") == "high"
        ]
    elif args.categories:
        categories = args.categories
    else:
        # Default to critical categories
        categories = [
            cat for cat, config in runner.category_configs.items()
            if config.get("critical", False)
        ]
    
    if not categories:
        logger.error("No test categories selected")
        sys.exit(1)
    
    logger.info(f"🚀 Running test categories: {', '.join(categories)}")
    
    # Configure options
    options = {
        "coverage": not args.no_coverage,
        "html_report": not args.no_html,
        "parallel": args.parallel
    }
    
    # Run tests
    results_data = runner.run_multiple_categories(categories, options)
    
    # Print individual category reports
    if args.verbose:
        for category, result in results_data["results"].items():
            runner.print_category_report(category, result)
    
    # Print analysis report
    runner.print_analysis_report(results_data["analysis"])
    
    # Save report
    if args.save_report:
        report_path = runner.save_report(
            results_data["results"],
            results_data["analysis"]
        )
        logger.info(f"📄 Detailed report saved to: {report_path}")
    
    # Exit with appropriate code
    if all(result.get("success", False) for result in results_data["results"].values()):
        logger.info("🎉 All selected test categories passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some test categories failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()