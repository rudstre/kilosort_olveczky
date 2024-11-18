from sampleToDateTime import sample_to_datetime, load_metadata

a = 30000000
metadata_path = "/Users/rudygelb-bicknell/Documents/python/spikeInterface/myRec/recording_metadata.json"

metadata = load_metadata(metadata_path)
ts = sample_to_datetime(a,metadata)

print(ts)