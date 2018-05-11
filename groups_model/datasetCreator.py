
import datasetWriter as dW

train_percent = 60
val_percent = 35
tag_non_erase = 60

print("\nDataset Creation Started\n")
print("\n-- Writing TF records...\n")
dW.run(train_percent, val_percent, tag_non_erase)
print("\nDataset Creation Finished\n")