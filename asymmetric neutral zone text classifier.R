# R script to accompany "A neutral zone classifier for three classes with an application to text mining"

# packages: word2vec, nnet, scales

# dataframe with 2 columns
# df$label: factor; true labels of the comments; 0,1,2
# df$comment: character; text data of interest
df
names(df) <- c("label","comment")
df$label <- factor(df$label,levels=c("0","1","2"))

# word2vec
# get word2vec model
# input vector comments
w2v.res <- word2vec::word2vec(tolower(df$comment))
# extract word embeddings matrix
emb <- as.matrix(w2v.res)
# get scores for each comment
d2v <- word2vec::doc2vec(w2v.res, newdata = tolower(df$comment), split = "[ .\n?!;,]")


# multinomial logistic regression with word2vec as predictors
# relevel to change the reference level
df$label <- relevel(df$label, ref="0")
# get indices of comments that have no score
na.ind <- which(is.na(d2v[,1]))
# new data frame for use in mult log reg, first col is true labels, rest is word embeddings
w2v <- data.frame(cbind(label=df$label,d2v))
# rename column
# run multinomial logistic regression models
log.fit <- nnet::multinom(label ~ ., data = w2v[-na.ind])
# get predicted probabilities
pred.probs <- predict(log.fit, type="prob")



# function to label based on L's
label.func <- function(p0,p1,L=rep(0,6)){
  l01 <- L[1]
  l02 <- L[2]
  l10 <- L[3]
  l12 <- L[4]
  l20 <- L[5]
  l21 <- L[6]
  in.func <- Vectorize(function(p0,p1,l01,l02,l10,l12,l20,l21){
    if((p0-(1-p0-p1)>l02 & (1-p0-p1)>p1) | (p0-p1>l01 & p1>(1-p0-p1))){
      out <- "0"
    } else if((p1-p0>l10 & p0>(1-p0-p1)) | (p1-(1-p0-p1)>l12 & (1-p0-p1)>p0)){
      out <- "1"
    } else if((1-p0-p1 - p0>l20 & p0>p1) | (1-p0-p1 - p1>l21 & p1>p0)){
      out <- "2"
    } else {out <- "l"} # l for neutral
  })
  in.func(p0,p1,l01,l02,l10,l12,l20,l21)
}


# six L
# loop to run grid search
all.comb <- do.call(expand.grid,list(seq(0,1,0.01),seq(0,1,0.01)))
true.labs <- df$label[-na.ind]
pos.labs <- c("0","1","2")
res.list <- list()
for(i1 in 1:3){
  res <- t(apply(all.comb,1,FUN = function(x){
    ind1 <- true.labs==pos.labs[-i1][1]
    ind2 <- true.labs==pos.labs[-i1][2]
    temp.L <- rep(0,6)
    temp.L[c(2*i1-1,2*i1)] <- x
    labs <- label.func(pred.probs[,1],pred.probs[,2],L=temp.L)
    # possibly add na's back in as neutrals if they exist
    # "l" for neutral
    if(length(na.ind)>0){
      pred.labs <- factor(c(labs,rep("l",length(na.ind))),levels=c("0","1","2","l"))
    }else{
      pred.labs <- factor(c(labs,levels=c("0","1","2","l")))
    }
    err1 <- sum(pred.labs[ind1]==pos.labs[i1])/sum(ind1)
    err2 <- sum(pred.labs[ind2]==pos.labs[i1])/sum(ind2)
    return(c(err1,err2))
  }))
  res <- cbind(all.comb,res)
  if(i1==1){
    colnames(res) <- c("L01","L02","err.1","err.2")
  } else if(i1==2){
    colnames(res) <- c("L10","L12","err.0","err.2")
  } else {
    colnames(res) <- c("L20","L21","err.0","err.1")
  }
  res.list[[i1]] <- res
  rm(i1)
}


# sort each predict zone by both error probabilities closest to alpha using objective function
# the top values are the chosen L's
alpha <- matrix(rep(0.1,6),byrow = T,nrow=3,ncol=2)
L <- rep(0,6L)
for(i1 in 1:3){
  # calculate objective function
  # using sum of absolute difference from target alpha as objective function
  res.list[[i1]]$obj <- abs(res.list[[i1]][,3]-alpha[i1,1]) + abs(res.list[[i1]][,4]-alpha[i1,2])
  # store 
  L[(2*i1-1):(2*i1)] <- unlist(res.list[[i1]][order(res.list[[i1]]$obj),][1,1:2])
  rm(i1)
}
names(L) <- c("L01","L02","L10","L12","L20","L21")


# table function for hard and neutral error rates by row, copies to clipboard
my.table <- function(true.labs,pred.labs,neut=T){
  temp.tab <- table(true.labs,pred.labs)
  l.ind <- rownames(temp.tab)
  err <- err.rt <- NULL
  for(i1 in 1:nrow(temp.tab)){
    ind <- match(l.ind[-i1],colnames(temp.tab))
    temp.err <- unname(temp.tab[i1,ind])
    err <- c(err,temp.err)
    temp.err.rt <- sum(temp.err)/sum(temp.tab[i1,])
    err.rt <- c(err.rt,temp.err.rt)
    rm(ind);rm(i1);rm(temp.err.rt);rm(temp.err)
  }
  err.rt <- round(c(err.rt,sum(err)/sum(temp.tab)),4)
  
  if(neut){
    neut.rt <- temp.tab[,"l"]/rowSums(temp.tab)
    neut.rt <- round(c(neut.rt, sum(temp.tab[,"l"])/sum(temp.tab)),4)
  }
  
  temp.tab <- round(prop.table(table(true.labs,pred.labs),1),4)
  if(any(colnames(temp.tab)=="l")){colnames(temp.tab)[colnames(temp.tab)=="l"] <- "Ne"}
  rownames(temp.tab) <- paste0("Tr=",rownames(temp.tab))
  colnames(temp.tab) <- paste0("Pr=",colnames(temp.tab))
  # out.tab <- addmargins(temp.tab)
  temp.tab <- rbind(temp.tab,Total=round(prop.table(addmargins(table(true.labs,pred.labs),1),1)[4,],4))
  out.tab <- temp.tab
  out.tab <- cbind(out.tab,Hard.Er=err.rt)
  if(!neut){colnames(out.tab)[ncol(out.tab)] <- "Error"}
  # write.table(out.tab, "clipboard", sep = "\t", row.names = TRUE, quote = FALSE)
  return(out.tab)
}

# predicted labels with six L neutral zone
labs <- label.func(p0=pred.probs[,1], p1=pred.probs[,2], L=L)
# table true vs predicted, row probabilities
my.table(true.labs,labs)


# plot 6 L neutral zone in posterior space
par(mar=c(5.1,5.1,4.1,2.1))
plot(1,type="n",xlim=0:1,ylim=0:1,xlab=expression(P*"("*hat(italic(C))*" = 0)"),ylab=expression(P*"("*hat(italic(C))*" = 1)"))

# find the (x,y) intercept of two lines
find.int <- function(a1,b1,a2,b2){
  # y = a1 + b1*x
  # y = a2 + b2*x
  # a1 + b1*x = a2 + b2*x
  x = (a2-a1)/(b1-b2)
  y = a1 + b1*x
  out <- c(x,y)
  names(out) <- c("x","y")
  return(out)
}
# function to shade classification regions in plot
add.L.color <- function(L){
  l01 <- L[1];l02 <- L[2];l10 <- L[3];l12 <- L[4];l20 <- L[5];l21 <- L[6]
  # intersections
  int.01 <- find.int(-l01,1,0.5,-0.5)
  int.02 <- find.int(1+l02,-2,0.5,-0.5)
  int.10 <- find.int(l10,1,1,-2)
  int.12 <- find.int(0.5+l12/2,-0.5,1,-2)
  int.20 <- find.int(1-l20,-2,0,1)
  int.21 <- find.int(0.5-l21/2,-0.5,0,1)
  int.0h <- find.int(-l01,1,1,-1)
  int.1h <- find.int(l10,1,1,-1)
  
  # shade classification regions
  col0 <- scales::alpha(1,0.2)
  polygon(x=c(1,int.01[1],int.0h[1],1),y=c(0,int.01[2],int.0h[2],0),col=col0,border=NA) # shade 01
  polygon(x=c(1,0.5+l02/2,int.02[1],1),y=c(0,0,int.02[2],0),col=col0,border=NA) # shade 02
  col1 <- scales::alpha(2,0.2)
  polygon(x=c(0,int.1h[1],int.10[1],0),y=c(1,int.1h[2],int.10[2],1),col=col1,border=NA) # shade 10
  polygon(x=c(0,int.12[1],0,0),y=c(1,int.12[2],0.5+l12/2,1),col=col1,border=NA) # shade 12
  col2 <- scales::alpha(3,0.2)
  polygon(x=c(0,0.5-l20/2,int.20[1],0),y=c(0,0,int.20[2],0),col=col2,border=NA) # shade 20
  polygon(x=c(0,0,int.21[1],0),y=c(0,0.5-l21/2,int.21[2],0),col=col2,border=NA) # shade 21
  
  # shade neutral zone
  coln <- scales::alpha(4,0.2)
  polygon(x=c(1/3,0.5,int.0h[1],int.01[1],1/3),y=c(1/3,0.5,int.0h[2],int.01[2],1/3),col=coln,border=NA) # neutral 01
  polygon(x=c(1/3,int.02[1],0.5+l02/2,0.5,1/3),y=c(1/3,int.02[2],0,0,1/3),col=coln,border=NA) # neutral 02
  polygon(x=c(1/3,0.5,int.1h[1],int.10[1],1/3),y=c(1/3,0.5,int.1h[2],int.10[2],1/3),col=coln,border=NA) # neutral 10
  polygon(x=c(1/3,int.12[1],0,0,1/3),y=c(1/3,int.12[2],0.5+l12/2,0.5,1/3),col=coln,border=NA) # neutral 12
  polygon(x=c(1/3,0.5,0.5-l20/2,int.20[1],1/3),y=c(1/3,0,0,int.20[2],1/3),col=coln,border=NA) # neutral 20
  polygon(x=c(1/3,0,0,int.21[1],1/3),y=c(1/3,0.5,0.5-l21/2,int.21[2],1/3),col=coln,border=NA) # neutral 21
}
add.L.color(L)

# add points to plot
points(x=pred.probs[labs!="l",1],y=pred.probs[labs!="l",2],col=true.labs[labs!="l"])
points(x=pred.probs[labs=="l",1],y=pred.probs[labs=="l",2],col=scales::alpha(true.labs[labs=="l"],1))
abline(1,-1)
# add legend to plot
legend("topright",legend=c(expression(italic(C)*" = 0"),expression(italic(C)*" = 1"),
                           expression(italic(C)*" = 2"),expression(hat(italic(C))*" = 0"),
                           expression(hat(italic(C))*" = 1"),expression(hat(italic(C))*" = 2"),
                           expression(hat(italic(C))*" = N")),
       pch=c(1,1,1,15,15,15,15),col=c(1,2,3,scales::alpha(1,0.2),scales::alpha(2,0.2),
                                      scales::alpha(3,0.2),scales::alpha(4,0.2)))


# function to add L borders to plot
add.Ls <- function(L){
  # lpo,lpn,lop,lon,lnp,lno
  l01 <- L[1];l02 <- L[2];l10 <- L[3];l12 <- L[4];l20 <- L[5];l21 <- L[6]
  # intersections
  int.01 <- find.int(-l01,1,0.5,-0.5)
  int.02 <- find.int(1+l02,-2,0.5,-0.5)
  int.10 <- find.int(l10,1,1,-2)
  int.12 <- find.int(0.5+l12/2,-0.5,1,-2)
  int.20 <- find.int(1-l20,-2,0,1)
  int.21 <- find.int(0.5-l21/2,-0.5,0,1)
  int.0h <- find.int(-l01,1,1,-1)
  int.1h <- find.int(l10,1,1,-1)
  
  # add natural border dotted lines
  lines(x=c(1/3,0.5),y=c(1/3,0),lty=3)
  lines(x=c(1/3,0.5),y=c(1/3,0.5),lty=3)
  lines(x=c(1/3,0),y=c(1/3,0.5),lty=3)
  
  lines(c(0,int.12[1]),c(0.5+l12/2,int.12[2]),lwd=1)
  lines(c(int.10[1],int.1h[1]),c(int.10[2],int.1h[2]),lwd=1)
  lines(c(0,int.21[1]),c(0.5-l21/2,int.21[2]),lwd=1)
  lines(c(int.20[1],0.5-l20/2),c(int.20[2],0),lwd=1)
  lines(c(int.01[1],int.0h[1]),c(int.01[2],int.0h[2]),lwd=1)
  lines(c(int.02[1],0.5+l02/2),c(int.02[2],0),lwd=1)
  
  # add text labels for each L
  text(x=int.0h[1],y=int.0h[2],labels=bquote(L["01"]*"="*.(l01)), adj=c(-0.2,0.2),cex=0.8) # L01
  mtext(text=bquote(L["02"]*"="*.(l02)), side=1, at=0.5+l02/2, line=0.4,cex=0.8) # L02
  mtext(text=bquote(L["12"]*"="*.(l12)), side=2, at=0.5+l12/2, line=0.4, las=1,cex=0.8) # L10
  text(x=int.1h[1],y=int.1h[2],labels=bquote(L["10"]*"="*.(l10)), adj=c(-0,-0.6),cex=0.8) # L12
  mtext(text=bquote(L["21"]*"="*.(l21)), side=2, at=0.5-l21/2, line=0.4, las=1,cex=0.8) # L20
  mtext(text=bquote(L["20"]*"="*.(l20)), side=1, at=0.5-l20/2, line=0.4,cex=0.8) # L21
  
  # 6 L drawing lines
  if(!isTRUE(all.equal(int.12,int.10))){lines(c(max(int.12[1],int.10[1]),min(int.12[1],int.10[1])),c(min(int.12[2],int.10[2]),max(int.12[2],int.10[2])),lwd=1)}
  if(!isTRUE(all.equal(int.21,int.20))){lines(c(max(int.21[1],int.20[1]),min(int.21[1],int.20[1])),c(max(int.21[2],int.20[2]),min(int.21[2],int.20[2])),lwd=1)}
  if(!isTRUE(all.equal(int.01,int.02))){lines(c(min(int.01[1],int.02[1]),max(int.01[1],int.02[1])),c(max(int.01[2],int.02[2]),min(int.01[2],int.02[2])),lwd=1)}
}

add.Ls(L)

