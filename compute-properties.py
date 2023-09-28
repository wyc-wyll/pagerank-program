from google.cloud import storage
import numpy as np
import statistics
import pdb
import concurrent.futures 
from html.parser import HTMLParser

class HW2HTMLParser(HTMLParser):

    destinations = []

    def __init__(self, *, convert_charrefs: bool = True) -> None:
        super().__init__(convert_charrefs=convert_charrefs)
        self.destinations = []

    def handle_starttag(self, tag, attrs):
        for attr in attrs:
            if (attr[0] == "href"):
                self.destinations.append(int(attr[1].split(".")[0]))

    def getDestinations(self):
        return self.destinations

client = storage.Client.create_anonymous_client()

outgoing = []
pageranks = [1/10000 for i in range(10000)]
new_pageranks = [1/10000 for i in range(10000)]
pageranks = np.array(pageranks)
new_pageranks = np.array(new_pageranks)

bucket_name = "hw2-ds561"
try:
    bucket = client.bucket(bucket_name)
except google.api_core.exceptions.NotFound:
    print("Bucket not found")

rows = 10000 
cols = 10000

matrix = [[0 for k in range(cols)] for i in range(rows)]

def download_file(blob):
    content = blob.download_as_text()
    return content

def compute(source, destination_list):
    for destination in destination_list:
        matrix[source][destination] += 1

def process_html_content(content):
    parser = HW2HTMLParser()
    parser.feed(content)
    return parser.destinations

def read_files():
    blob_list = list(bucket.list_blobs())
    results = None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        inputs = blob_list
        results = executor.map(download_file, inputs) # should be a list of html contents


    for source, content in zip(inputs, results):
        compute(int(source.name.split(".")[0]), process_html_content(content=content))

def get_single_pagerank(i):
    global matrix
    global outgoing
    global pageranks
    variable = 0
    for k in range(10000):
        if (matrix[k][i] != 0):
            variable += (pageranks[k]/outgoing[k])
    return 0.15 + 0.85 * variable

def compute_pagerank():
    global matrix
    global pageranks
    global outgoing

    outgoing = np.array(outgoing)

    get_pagerank = np.vectorize(get_single_pagerank)

    matrix = np.array(matrix)


    while True:
        print("Calculating pagerank list")
        new_pageranks[np.arange(10000)] = get_pagerank(i=np.arange(10000))
        print(new_pageranks)
        print(pageranks)
        if ((sum(new_pageranks) - sum(pageranks))/sum(pageranks) <= 0.005):
            pageranks = new_pageranks
            print("Updating...")
            break
        else:
            print("Updating and Continue...")
            pageranks = new_pageranks
    
    return pageranks


def get_outgoing(matrix):
    return [sum(i) for i in matrix]

def get_incoming(matrix):
    result = [0 for i in range(10000)]
    for i in range(10000):
        for k in range(10000):
            result[i] += matrix[k][i]
    return result

def main():
    global matrix
    global outgoing
    read_files()

    outgoing_count = get_outgoing(matrix=matrix)
    outgoing = outgoing_count
    incoming_count = get_incoming(matrix=matrix)

    pagerank_result = compute_pagerank()

    top_5 = np.argpartition(pagerank_result, -5)[-5:]

    # median
    median_outgoing = statistics.median(outgoing_count)
    median_incoming = statistics.median(incoming_count)
    
    # average
    avg_outgoing = statistics.mean(outgoing_count)
    avg_incoming = statistics.mean(incoming_count)
    
    # max
    max_outgoing = max(outgoing_count)
    max_incoming = max(incoming_count)
    
    # min
    min_outgoing = min(outgoing_count)
    min_incoming = min(incoming_count)

    # quintiles -> outgoing
    quintile_1st_outgoing = np.quantile(outgoing_count, 0.2)
    quintile_2nd_outgoing = np.quantile(outgoing_count, 0.4)
    quintile_3rd_outgoing = np.quantile(outgoing_count, 0.6)
    quintile_4th_outgoing = np.quantile(outgoing_count, 0.8)
    quintile_5th_outgoing = np.quantile(outgoing_count, 1.0)

    # quintiles -> incoming
    quintile_1st_incoming = np.quantile(incoming_count, 0.2)
    quintile_2nd_incoming = np.quantile(incoming_count, 0.4)
    quintile_3rd_incoming = np.quantile(incoming_count, 0.6)
    quintile_4th_incoming = np.quantile(incoming_count, 0.8)
    quintile_5th_incoming = np.quantile(incoming_count, 1.0)

    print("Outgoing average: " + str(avg_outgoing))
    print("Outgoing median: " + str(median_outgoing))
    print("Outgoing max: " + str(max_outgoing))
    print("Outgoing min: " + str(min_outgoing))
    print("1st Outgoing Quintile: " + str(quintile_1st_outgoing))
    print("2nd Outgoing Quintile: " + str(quintile_2nd_outgoing))
    print("3rd Outgoing Quintile: " + str(quintile_3rd_outgoing))
    print("4th Outgoing Quintile: " + str(quintile_4th_outgoing))
    print("5th Outgoing Quintile: " + str(quintile_5th_outgoing))

    print("Incoming average: " + str(avg_incoming))
    print("Incoming median: " + str(median_incoming))
    print("Incoming max: " + str(max_incoming))
    print("Incoming min: " + str(min_incoming))
    print("1st Incoming Quintile: " + str(quintile_1st_incoming))
    print("2nd Incoming Quintile: " + str(quintile_2nd_incoming))
    print("3rd Incoming Quintile: " + str(quintile_3rd_incoming))
    print("4th Incoming Quintile: " + str(quintile_4th_incoming))
    print("5th Incoming Quintile: " + str(quintile_5th_incoming))

    print("Graph Adjacency list: ")
    print(np.array(matrix))

    print("Pagerank ordered by html filename value (1,2,3,...,9999): ")
    print(pagerank_result)

    print("top 5 ranked pages: ")
    print(top_5)

if __name__ == "__main__":
    main()