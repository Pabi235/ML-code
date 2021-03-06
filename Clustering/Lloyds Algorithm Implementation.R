dataset=read.csv("SeedsData.csv")  # I used the seeds data set from the  UCI machine learning repository 
                                   # https://archive.ics.uci.edu/ml/datasets/seeds

# Simple Implementation of Lloyds Algorithm for K means clustering.

k_means=function(data,k){
  
  #choose intial centroids. and label vector
  n_obs=nrow(data)
  n_features=ncol(data)
  intit_centroid=data[sample(1:nrow(data), k,
                             replace=FALSE),]
  label_vector= rep(0,n_obs)
  label_vector_final= rep(0,n_obs)
# continue_clustering=TRUE
  
  
  for(i in 1:k){
    assign(paste("cluster_", i, sep = ""), data.matrix(intit_centroid[i,])) 
    assign(paste("cluster_", i, sep = ""), t(get(paste("cluster_", i, sep = "")))) 
  }
  
  #counter variables
  old_centroid=intit_centroid
  iterations=0
  max_iterations=30
  
  
  # while (continue_search(iterations,old_centroid)==TRUE) {
  # while((continue_clustering==TRUE)&(iterations < max_iterations)){
  while(iterations < max_iterations){
    # for observation x_i calculate the squared distance between centroid k and x_i
    #do this for every centroid i \in 1:k
    #assign x_i to n \in 1:k such that dist(x_i,mean_k is minimised)
    for (j in 1:n_obs){
      cent_dist=rep(0,k)
      for (i in 1:k){
        cent_dist[i]=sum((data[j,]-old_centroid[i,])^2)
      }
      
      assign(paste("cluster_", which.min(cent_dist), sep = ""),
             rbind(get(paste("cluster_", which.min(cent_dist), sep = ""))
                   ,t(data.matrix(data[j,]))))
      label_vector[j]=which.min(cent_dist)
    }        
    
    #calculate the new means of the clusters after assigns all observations
    #to appropriate clusters
    for(i in 1:k){
      assign(paste("cluster_", i, sep = ""), t(data.matrix(apply(get(paste("cluster_",i , sep = ""))                                             ,2,mean))))
      
    }  
    if(!identical(label_vector,label_vector_final)){
      continue_clustering=TRUE
      label_vector_final=label_vector
    } else {continue_clustering=FALSE}
    iterations=iterations+1
    #update the centroid matrix to the new centroid vectors
    for(j in 1:k){
      old_centroid[j,]=get(paste("cluster_",j , sep = ""))
    }
    
  }  #repeat
  
  #statistic creation for kmeans output
  
  #create within cluster variance variables
  for (j in 1:k){
    assign(paste("clusvar_", j, sep = ""), 0) 
  } 
  
  for (j in 1:n_obs){
    for (z in 1:k){
      if (label_vector_final[j]==z){
        assign(paste("clusvar_",z,sep=""),
               get(paste("clusvar_",z,sep=""))+sum((data[j,]-get(paste("cluster_",z,sep="")))^2))
      }
    }
  }
  within_vect=c(clusvar_1)
  for (c in 2:k){
    within_vect=c(within_vect,get(paste("clusvar_",c,sep = "")))
    
  }
  
  
  total_var=sum(diag(var(data)))
  #to be used in contructing plot for elbow method.
  
  distances=matrix(0,ncol = 210,nrow = 1)
  for (j in 1:n_obs){
    if (label_vector_final[j]==1){
      distances[j]=sum((data[j,]-cluster_1)^2)
    }else if (label_vector_final[j]==2){
      distances[j]=sum((data[j,]-cluster_2)^2)
    }else{
      distances[j]=sum((data[j,]-cluster_3)^2)
    }
    
  }
  mean_dist=mean(distances)
  return (list(centroidmatirx=old_centroid,obs_labels=label_vector,
               mean_distance=mean_dist, withinclusvar=within_vect))
}

k_means(dataset,3)  #give it a try.


