import helper
import glob
# helper.test_data("500_bst/", "analysis/500_bst/")
# helper.test_data("700_bst/", "analysis/700_bst/")
# helper.test_data("900_bst/", "analysis/900_bst/")
# helper.test_data("1100_bst/", "analysis/1100_bst/")
# helper.test_data("1500_bst/", "analysis/1500_bst/")
# helper.test_data("1700_bst/", "analysis/1700_bst/")
# helper.test_data("1900_bst/", "analysis/1900_bst/")
# helper.test_data("2000_bst/", "analysis/2000_bst/")
# helper.test_data("500_bst_opening/", "analysis/500_bst_opening/")
# helper.test_data("400_bst_opening/", "analysis/400_bst_opening/")

paths = glob.glob("inference/*bst*")
for i, _ in enumerate(paths):
    paths[i] = paths[i].split('/')[1]

for path in paths:
    print(path)
    print()
    helper.test_data_per_class(path, "analysis/testing/")
    print()
    print()
    print()
