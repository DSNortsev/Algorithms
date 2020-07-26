#!/usr/bin/env python3

import sys
import re

'''
   Script accepts the follwing intervals format: "{8,10},{2,6},{1,3},{15,18}"
'''

def sort_merge_intervals(intervals):
  sorted_intervals = sorted( intervals, key=lambda x: x[0])
  index = 0
  for interval in sorted_intervals:
    if interval[0] > sorted_intervals[index][1]:
      index += 1
      sorted_intervals[index] = interval
    else:
      sorted_intervals[index] = [sorted_intervals[index][0], interval[1]]
  return sorted_intervals[:index+1]


def main():
    if  len(sys.argv) < 2:
       print('Please pass a list of intervals. For example: "{8,10},{2,6},{1,3},{15,18}"')
       sys.exit(0)

    intervals = [ list(map(int, i.split(','))) for i in re.findall(r'{(.*?)}', sys.argv[1]) ]

    if len(intervals) == 0:
        print('Number of valid intervals is 0.\n', \
	      'Please pass a list of intervals. For example: "{8,10},{2,6},{1,3},{15,18}"', sep='')
        sys.exit(0)
  
    merged_intervals  = sort_merge_intervals(intervals)

    for interval in merged_intervals:
        print(f'{interval[0]} - {interval[1]}')


if __name__ == '__main__':
    main()
