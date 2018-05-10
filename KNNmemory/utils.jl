# Partitions the list l into chunks of the length n
partition(list, n) = [list[i:min(i + n - 1,length(list))] for i in 1:n:length(list)]
