import numpy as np

# Train
PHQ9_mul_GT = [46, 19, 5, 4]
PHQ9_mul_RE = [45, 14, 5, 2]
cwa = [PHQ9_mul_RE[i]/PHQ9_mul_GT[i] for i in range(4)]
wa = sum(cwa) / len(cwa)
print("PHQ9 multiclass train: " + str(wa))

PHQ9_bin_GT = [46, 28]
PHQ9_bin_RE = [45, 27]
cwa = [PHQ9_bin_RE[i]/PHQ9_bin_GT[i] for i in range(2)]
wa = sum(cwa) / len(cwa)
print("PHQ9 binary train: " + str(wa))

GAD7_mul_GT = [42, 15, 9, 8]
GAD7_mul_RE = [41, 13, 7, 6]
cwa = [GAD7_mul_RE[i]/GAD7_mul_GT[i] for i in range(4)]
wa = sum(cwa) / len(cwa)
print("GAD7 multiclass train: " + str(wa))

GAD7_bin_GT = [42, 32]
GAD7_bin_RE = [41, 31]
cwa = [GAD7_bin_RE[i]/GAD7_bin_GT[i] for i in range(2)]
wa = sum(cwa) / len(cwa)
print("PHQ9 binary train: " + str(wa))
print("\n")

# Val
PHQ9_mul_GT = [8, 3, 1, 1]
PHQ9_mul_RE = [7, 1, 0, 0]
cwa = [PHQ9_mul_RE[i]/PHQ9_mul_GT[i] for i in range(4)]
wa = sum(cwa) / len(cwa)
print("PHQ9 multiclass val: " + str(wa))

PHQ9_bin_GT = [8, 5]
PHQ9_bin_RE = [6, 5]
cwa = [PHQ9_bin_RE[i]/PHQ9_bin_GT[i] for i in range(2)]
wa = sum(cwa) / len(cwa)
print("PHQ9 binary val: " + str(wa))

GAD7_mul_GT = [6, 4, 2, 1]
GAD7_mul_RE = [5, 0, 0, 0]
cwa = [GAD7_mul_RE[i]/GAD7_mul_GT[i] for i in range(4)]
wa = sum(cwa) / len(cwa)
print("GAD7 multiclass val: " + str(wa))

GAD7_bin_GT = [6, 7]
GAD7_bin_RE = [4, 5]
cwa = [GAD7_bin_RE[i]/GAD7_bin_GT[i] for i in range(2)]
wa = sum(cwa) / len(cwa)
print("PHQ9 binary val: " + str(wa))
print("\n")


# Test
PHQ9_mul_GT = [14, 6, 1, 1]
PHQ9_mul_RE = [12, 0, 0, 0]
cwa = [PHQ9_mul_RE[i]/PHQ9_mul_GT[i] for i in range(4)]
wa = sum(cwa) / len(cwa)
print("PHQ9 multiclass test: " + str(wa))

PHQ9_bin_GT = [14, 8]
PHQ9_bin_RE = [8, 3]
cwa = [PHQ9_bin_RE[i]/PHQ9_bin_GT[i] for i in range(2)]
wa = sum(cwa) / len(cwa)
print("PHQ9 binary test: " + str(wa))

GAD7_mul_GT = [12, 5, 2, 3]
GAD7_mul_RE = [8, 0, 0, 0]
cwa = [GAD7_mul_RE[i]/GAD7_mul_GT[i] for i in range(4)]
wa = sum(cwa) / len(cwa)
print("GAD7 multiclass test: " + str(wa))

GAD7_bin_GT = [12, 10]
GAD7_bin_RE = [8, 4]
cwa = [GAD7_bin_RE[i]/GAD7_bin_GT[i] for i in range(2)]
wa = sum(cwa) / len(cwa)
print("PHQ9 binary test: " + str(wa))
print("\n")


# Val-Test
PHQ9_mul_GT = np.array([8, 3, 1, 1]) + np.array([14, 6, 1, 1])
PHQ9_mul_RE = np.array([7, 1, 0, 0]) + np.array([12, 0, 0, 0])
cwa = [PHQ9_mul_RE[i]/PHQ9_mul_GT[i] for i in range(4)]
wa = sum(cwa) / len(cwa)
print("PHQ9 multiclass val+test: " + str(wa))

PHQ9_bin_GT = np.array([8, 5]) + np.array([14, 8])
PHQ9_bin_RE = np.array([6, 5]) + np.array([8, 3])
cwa = [PHQ9_bin_RE[i]/PHQ9_bin_GT[i] for i in range(2)]
wa = sum(cwa) / len(cwa)
print("PHQ9 binary val+test: " + str(wa))

GAD7_mul_GT = np.array([6, 4, 2, 1]) + np.array([12, 5, 2, 3])
GAD7_mul_RE = np.array([5, 0, 0, 0]) + np.array([8, 0, 0, 0])
cwa = [GAD7_mul_RE[i]/GAD7_mul_GT[i] for i in range(4)]
wa = sum(cwa) / len(cwa)
print("GAD7 multiclass val+test: " + str(wa))

GAD7_bin_GT = np.array([6, 7]) + np.array([12, 10])
GAD7_bin_RE = np.array([4, 5]) + np.array([8, 4])
cwa = [GAD7_bin_RE[i]/GAD7_bin_GT[i] for i in range(2)]
wa = sum(cwa) / len(cwa)
print("PHQ9 binary val+test: " + str(wa))
print("\n")



