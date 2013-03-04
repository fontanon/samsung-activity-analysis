# Final Loan Analysis
# ===================

# Perform download of source files 
download.file("http://spark-public.s3.amazonaws.com/dataanalysis/samsungData.rda", destfile="data/samsungData.rda")
dateDownloaded <- date()
load("data/samsungData.rda")