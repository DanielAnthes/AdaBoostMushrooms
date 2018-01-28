graphics.off()
rm(list = ls())

require(ggplot2)
require(DescTools)

data = c(
  'Mushrooms/results.csv',
  'Senate/results.csv',
  'Connect4/results.csv'
)[1]

results = read.table(data, header = T, sep = ',')

to_num_list <- function(vector) {
  string_vector = as.character(vector)
  result = list()
  for(i in seq(length(string_vector))){
    result[[i]] = as.numeric(unlist(strsplit(string_vector[i], split = ",")))
  }
  return(result)
}

for(i in seq(3, 4)){
  results[[i]] = to_num_list(results[[i]])
}

colour_space = c('rgb', 'cmy')[2]
colours = c()
if(colour_space == "rgb"){
  colours = c("red", "green", "blue", "red3", "green4", "blue2")
} else if(colour_space == "cmy"){
  colours = c("cyan", "magenta", "yellow2", "cyan4", "magenta3", "yellow4")
} else {
  return("incorrect colour space: rgb/cmy")
}

upper = c()
lower = c()
mean = c()
set = c()
step = c()
col = c()

for(step_nr in results$X){
  mean_ci = MeanCI(subset(results, X == step_nr)$boost.test.err[[1]], conf.level = 0.95)
  upper = c(upper, mean_ci[[3]])
  lower = c(lower, mean_ci[[2]])
  mean = c(mean, mean_ci[[1]])
  set = c(set, 'test')
  step = c(step, step_nr)
  col = c(col, colours[1])
}
for(step_nr in results$X){
  mean_ci = MeanCI(subset(results, X == step_nr)$boost.train.err[[1]], conf.level = 0.95)
  upper = c(upper, mean_ci[[3]])
  lower = c(lower, mean_ci[[2]])
  mean = c(mean, mean_ci[[1]])
  set = c(set, 'train')
  step = c(step, step_nr)
  col = c(col, colours[2])
}

    
bands_df = data.frame(
  upper,
  lower,
  mean,
  set,
  step,
  col
)
ggplot(bands_df) + 
         geom_ribbon(data=subset(bands_df, set=='train'), aes(ymin = lower, ymax = upper, x = step),
                     fill = colours[2], alpha = 0.4) + 
         geom_line(data=subset(bands_df, set=='train'), color = colours[5], aes(x = step, y = mean), size = 1.2) +
         geom_ribbon(data=subset(bands_df, set=='test'), aes(ymin = lower, ymax = upper, x = step),
                     fill = colours[1], alpha = 0.4) + 
         geom_line(data=subset(bands_df, set=='test'), color = colours[4], aes(x = step, y = mean), size = 1.2)


flatten_df <- function(dframe){
  result = c()
  for (col in 4:ncol(dframe)){
    for (vec in 1:10){
      result = c(result, dframe[[col]][[vec]])
    }
  }
  return(result)
}

{
  weights <- data.frame(
              val = c(flatten_df(bot1_rate0), flatten_df(bot1_rate01),
                      flatten_df(bot1_rate02), flatten_df(bot3_rate02),
                      flatten_df(bot5_rate02)),
              step = rep(seq(0, 19999, by=1000), 4*11*10),
              weight = rep(c(rep("ll", 10*20), rep("lr", 10*20), rep("rl", 10*20), rep("rr", 10*20)), 11),
              bots = c(rep("1", 10*20*4*3), rep("3", 10*20*4*3), rep("5", 10*20*4*5)),
              rate = c(rep("0", 10*20*4), rep("0.1", 10*20*4), rep("0.2", 10*20*4*9)),
              agent_nr = c(rep(rep(1:10, each=20), 4), rep(rep(11:20, each=20), 4), rep(rep(21:30, each=20), 4),
                           rep(rep(31:40, each=20), 4), rep(rep(41:50, each=20), 4), rep(rep(51:60, each=20), 4),
                           rep(rep(61:70, each=20), 4), rep(rep(71:80, each=20), 4), rep(rep(81:90, each=20), 4),
                           rep(rep(91:100, each=20), 4), rep(rep(101:110, each=20), 4))
  )

  line_plot <- function(dframe){
    return(ggplot(data = dframe, aes(x = step, y = val)) +
             geom_line(data=subset(dframe, bots==5), aes(colour=bots, group=agent_nr), size=1, alpha=.4) +
             geom_line(data=subset(dframe, bots==3), aes(colour=bots, group=agent_nr), size=1, alpha=.4) +
             geom_line(data=subset(dframe, bots==1), aes(colour=bots, group=agent_nr), size=1, alpha=.4) +
             scale_colour_hue(c=300, l=60))
  }
  
  bands_plot <- function(dframe, vary_attribute, colour_space){
    colours = c()
    if(colour_space == "rgb"){
      colours = c("red", "green", "blue", "red3", "green4", "blue2")
    } else if(colour_space == "cmy"){
      colours = c("cyan", "magenta", "yellow2", "cyan4", "magenta3", "yellow4")
    } else {
      return("incorrect colour space: rgb/cmy")
    }
    if(vary_attribute != "bots" & vary_attribute != "learn rate"){
      return("incorrect attribute to vary: bots/learn rate")
    }
    
    upper = c()
    lower = c()
    mean = c()
    bots = c()
    step = c()
    rate = c()
    rate = c()
    
    if(vary_attribute == "bots"){
      for(bot_nr in c(1, 3, 5)){
        for(step_nr in seq(0, 19999, by = 1000)){
          mean_ci = MeanCI(subset(dframe, bots == bot_nr & step == step_nr)$val, conf.level = 0.95)
          upper = c(upper, mean_ci[[3]])
          lower = c(lower, mean_ci[[2]])
          mean = c(mean, mean_ci[[1]])
          bots = c(bots, bot_nr)
          step = c(step, step_nr)
        }
      }
      bands_df = data.frame(
          upper,
          lower,
          mean,
          bots,
          step
      )
      return(ggplot() + 
               geom_ribbon(data=subset(bands_df, bots==1), aes(ymin = lower, ymax = upper, x = step),
                           fill = colours[3], alpha = 0.4) + 
               geom_line(data=subset(bands_df, bots==1), color = colours[6], aes(x = step, y = mean), size = 1.2) +
               geom_ribbon(data=subset(bands_df, bots==3), aes(ymin = lower, ymax = upper, x = step),
                           fill = colours[2], alpha = 0.4) + 
               geom_line(data=subset(bands_df, bots==3), color = colours[5], aes(x = step, y = mean), size = 1.2) +
               geom_ribbon(data=subset(bands_df, bots==5), aes(ymin = lower, ymax = upper, x = step),
                           fill = colours[1], alpha = 0.4) +
               geom_line(data=subset(bands_df, bots==5), color = colours[4], aes(x = step, y = mean), size = 1.2)
      )
    } else if(vary_attribute == "learn rate"){
      for(learn_rate in c(0, 0.1, 0.2)){
        for(step_nr in seq(0, 19999, by = 1000)){
          mean_ci = MeanCI(subset(dframe, step == step_nr & rate == learn_rate)$val, conf.level = 0.95)
          upper = c(upper, mean_ci[[3]])
          lower = c(lower, mean_ci[[2]])
          mean = c(mean, mean_ci[[1]])
          rate = c(rate, learn_rate)
          step = c(step, step_nr)
        }
      }
      bands_df = data.frame(
        upper,
        lower,
        mean,
        rate,
        step
      )
      return(ggplot() + 
               geom_ribbon(data=subset(bands_df, rate==0.2), aes(ymin = lower, ymax = upper, x = step),
                          fill = colours[1], alpha = 0.4) + 
               geom_line(data=subset(bands_df, rate==0.2),
                          color = colours[4], aes(x = step, y = mean), size = 1.2) +
               geom_ribbon(data=subset(bands_df, rate==0.1), aes(ymin = lower, ymax = upper, x = step),
                          fill = colours[2], alpha = 0.4) + 
               geom_line(data=subset(bands_df, rate==0.1),
                          color = colours[5], aes(x = step, y = mean), size = 1.2) +
               geom_ribbon(data=subset(bands_df, rate==0.0), aes(ymin = lower, ymax = upper, x = step),
                          fill = colours[3], alpha = 0.4) +
               geom_line(data=subset(bands_df, rate==0.0),
                          color = colours[6], aes(x = step, y = mean), size = 1.2) + 
               scale_fill_manual(values=c("blue", "red", "green"))
      )
    }
  }
} # weight formatting stuff

{
  weights_rate_ll = subset(weights, bots == 1 & weight == "ll")
  weights_rate_lr = subset(weights, bots == 1 & weight == "lr")
  weights_rate_rl = subset(weights, bots == 1 & weight == "rl")
  weights_rate_rr = subset(weights, bots == 1 & weight == "rr")
  
  bands_plot(weights_rate_ll, "learn rate", "rgb") +
    labs(x = "simulation step", y = "weight value", title = "weight: LL")
  bands_plot(weights_rate_lr, "learn rate", "rgb") +
    labs(x = "simulation step", y = "weight value", title = "weight: LR")
  bands_plot(weights_rate_rl, "learn rate", "rgb") +
    labs(x = "simulation step", y = "weight value", title = "weight: RL")
  bands_plot(weights_rate_rr, "learn rate", "rgb") +
    labs(x = "simulation step", y = "weight value", title = "weight: RR")
  
  weights_bots_ll = subset(weights, rate == 0.2 & weight == "ll")
  weights_bots_lr = subset(weights, rate == 0.2 & weight == "lr")
  weights_bots_rl = subset(weights, rate == 0.2 & weight == "rl")
  weights_bots_rr = subset(weights, rate == 0.2 & weight == "rr")
  
  bands_plot(weights_bots_ll, "bots", "cmy") +
    labs(x = "simulation step", y = "weight value", title = "weight: LL")
  bands_plot(weights_bots_lr, "bots", "cmy") +
    labs(x = "simulation step", y = "weight value", title = "weight: LR")
  bands_plot(weights_bots_rl, "bots", "cmy") +
    labs(x = "simulation step", y = "weight value", title = "weight: RL")
  bands_plot(weights_bots_rr, "bots", "cmy") +
    labs(x = "simulation step", y = "weight value", title = "weight: RR")
} # weight plotting stuff

clusters = data.frame(
            nr_of_clusters = c(bot1_rate0$nr_of_clusters, bot1_rate01$nr_of_clusters,
                               bot1_rate02$nr_of_clusters, bot3_rate02$nr_of_clusters,
                               bot5_rate02$nr_of_clusters),
            mean_cluster_size = c(bot1_rate0$mean_cluster_size, bot1_rate01$mean_cluster_size,
                                  bot1_rate02$mean_cluster_size, bot3_rate02$mean_cluster_size,
                                  bot5_rate02$mean_cluster_size),
            perc_in_cluster = c(bot1_rate0$perc_in_cluster, bot1_rate01$perc_in_cluster,
                                bot1_rate02$perc_in_cluster, bot3_rate02$perc_in_cluster,
                                bot5_rate02$perc_in_cluster),
            bots = c(rep(1, 30), rep(3, 10), rep(5, 10)),
            rate = c(rep(0.0, 10), rep(0.1, 10), rep(0.2, 30))
)

cluster_plot <- function(dframe, x_var, y_var){
  bar_width = 0
  if(x_var == "bots"){
    x_var = dframe$bots
    bar_width = 0.5
  } else if(x_var == "learn rate"){
    x_var = dframe$rate 
    bar_width = 0.025
  } else {
    return("incorrect attribute to vary: bots/learn rate")
  }
  
  if(y_var == "cluster nr"){
    y_var = 1
  } else if(y_var == "mean size"){
    y_var = 2
  } else if(y_var == "perc in cluster"){
    y_var = 3
  } else {
    return("incorrecy y variable: cluster nr/mean size/perc in cluster")
  }
  
  return(ggplot(dframe, aes(x_var, dframe[[y_var]])) + stat_summary(fun.y = mean, geom = "line") + stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = bar_width) + stat_summary(fun.y = mean, geom = "point", color = "Red")
  )
}


clusters_bots = subset(clusters, rate == 0.2)

cluster_plot(clusters_bots, "bots", "cluster nr") +
  labs(x = "bots", y = "nr. of clusters", title = "learning rate = 0.2")
cluster_plot(clusters_bots, "bots", "perc in cluster") +
  labs(x = "bots", y = "% of blocks in clusters", title = "learning rate = 0.2")
cluster_plot(subset(clusters_bots, nr_of_clusters != 0), "bots", "mean size") +
  labs(x = "bots", y = "mean cluster size", title = "learning rate = 0.2")

clusters_rate = subset(clusters, bots == 1)

cluster_plot(clusters_rate, "learn rate", "cluster nr") +
  labs(x = "learning rate", y = "nr. of clusters", title = "bots = 1")
cluster_plot(clusters_rate, "learn rate", "perc in cluster") +
  labs(x = "learning rate", y = "% of blocks in clusters", title = "bots = 1")
cluster_plot(subset(clusters_rate, nr_of_clusters != 0), "learn rate", "mean size") +
  labs(x = "learning rate", y = "mean cluster size", title = "bots = 1")
