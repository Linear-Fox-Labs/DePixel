import unittest
import os
import sys
 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def run_tests(): 
    loader = unittest.TestLoader()
    start_dir = os.path.join(project_root, 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
 
    return len(result.failures) + len(result.errors)

if __name__ == '__main__':
    failures_and_errors = run_tests()
    sys.exit(failures_and_errors)