import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import networkx as nx

original_json_file = "./tmp/raw/Video_Games_5.json"
stripped_json_file = "./tmp/raw/Modified_Video_Games_5.json"
output_csv_file = "./tmp/raw/Modified_Video_Games_5.csv"

def main():
  print("Starting analysis")
  print("Reading json file")
  print("Stripping unnecessary data and converting to proper json...")
  filterData(original_json_file, stripped_json_file)

  #Convert json to csv
  #data = pd.read_json("Modified_Video_Games_5.json")
  #data = pd.read_json("test.json")
  print("Converting json to csv...")
  convertJsonToCSV(stripped_json_file, output_csv_file)

  # print("Finished reading json file")
  # print("{0} Reviews, {1} Columns".format(data.shape[0] ,data.shape[1]))
  # #print(data.head())
  # group_by_reviewer = data.groupby(['reviewerID'])['reviewerID'].count()
  # #print(group_by_reviewer)
  # print("# Unique Reviewers: {0}".format(len(group_by_reviewer)))
  # #print("# Unique Reviewers: {0}".format(len(data['reviewerID'].unique())))
  # print("# Unique Items: {0}".format(len(data['asin'].unique())))
  # print("Average number of reviews: {0}".format(group_by_reviewer.mean()))
  # print("Average review rating: {0}".format(data['overall'].mean()))

  # # Users are connected if they rate the same item
  # graph(data)
  print("Finished analysis")


def convertJsonToCSV(json_file, csv_file):
  with open(json_file, encoding='utf-8-sig') as f_input:
    df = pd.read_json(f_input)

  df.to_csv(csv_file, encoding='utf-8', index=False)

# Function to remove unnecessary data from original file. (Also mangles it to become true json)
# Saves filtered data to new json file
def filterData(file_name, to_file_name):
  data = "ERROR"
  with open(file_name, 'r') as f:
    json_stripped = []
    for line in f:
      json_line = json.loads(line.strip())
      #print("Element: " + str(json_line))
      json_line.pop('reviewText', None)
      json_line.pop('summary', None)
      #print("\n\nRemoved review text and summary")
      #print("Element: " + str(json_line))

      json_stripped.append(json.dumps(json_line))

    data = ",\n".join(json_stripped)
    print('Stripped out useless data')

  with open(to_file_name, 'w') as f:
    f.write("[\n")
    f.write(data)
    f.write("]\n")


# Nodes represent users. Draw a line between users if they have rated thte same item
def graph(df):
  G = nx.Graph()

  #items = dataFrame['asin'].unique()
  #items.sort()
  total_length = df.shape[0]
  dataFrame = df.sort_values("asin", ascending=False)


  for index, (_, row) in enumerate(dataFrame.iterrows()):
    if index % 1000 == 0:
      print("{0:.2f}% Complete: {1}/{2}".format(float(index)/float(total_length)*100.0, index, total_length))
    #print("-------------------------------")
    #print(row)
    row1_item = row['asin']
    reviewer1 = row['reviewerID']
    index2 = index
    while True:
      if index2 >= total_length:
        break
      row2 = dataFrame.iloc[index2]
      #print(row2)
      #print("-------------------------------")

      if row2['asin'] != row1_item:
        #print("\nItems don't match between reviwer1 and reviwer2\n")
        break

      reviewer2 = row2['reviewerID']
      if reviewer1 != reviewer2:
        #print('%s and %s' % (author, author2))
        if reviewer1 in G.adj and reviewer2 in G[reviewer1]:
          G[reviewer1][reviewer2]['count'] += 1
        else:
          #print("Adding edge")
          G.add_edges_from([(reviewer1, reviewer2, {'count':1})])
          #G.add_edges_from([(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})])
      index2 = index2 + 1

    #return # Return after 1 item

  print("Number of nodes: %d" % G.number_of_nodes())
  print("Number of edges: %d" % G.number_of_edges())
  
  # POSSIBLE ANALYSIS: Clustering Coefficient, Pagerank, Diameter, Closeness, Betweeness, HITS

  # ISSUE: Graph is not connected
  #print("DIAMETER: ")
  #print(nx.diameter(G))

  # CLUSTERING COEFFICIENT --------------------------------------
  print("CLUSTERING COEFFICIENT (TRANSITIVITY): ")
  print(nx.transitivity(G))


  # PAGE RANK --------------------------------------
  pr = nx.pagerank(G, alpha=0.9)

  pr_list = []
  for key, value in pr.items():
    temp = [key,value]
    pr_list.append(temp)
  
  pagerank_df = pd.DataFrame(pr_list)
  pagerank_df.to_csv('pagerank.csv')

  
  print("PAGE RANK: ")
  print("Saved to pagerank.csv")
  #print(pr)

  ##### Draw the graph -------------------------------
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos, node_size=3)
  nx.draw_networkx_edges(G, pos, alpha=0.4)
  plt.savefig('cs_graph.png')

  plt.show()

  ##### Display a histogram -------------------------------
  degree_histogram(G)
  

def degree_histogram(G):
  # Get degree sequence
  ds = sorted([d for n, d in G.degree()], reverse=True)  
  print("Degree sequence " + str(ds))

  # Count degrees
  degreeCount = collections.Counter(ds) 
  print(degreeCount)

  # Split into degree, count
  deg, count = zip(*degreeCount.items())
  print(deg)
  print(count)

  # Plot the data!
  figure, axis = plt.subplots() 
  plt.bar(deg, count, width=0.80, color='green')

  plt.title("User Degree Histogram")
  plt.ylabel("Count")
  plt.xlabel("Degree")
  axis.set_xticks([0,10,20,30,40,50,60,70,80])
  axis.set_xticklabels([0,10,20,30,40,50,60,70,80])

  plt.savefig('degrees_graph.png')

  plt.show()

if __name__ == "__main__":
  main()