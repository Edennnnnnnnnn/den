
// Eden Zhou 20201026


using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class KMeans
{
    const int K = 6;    //TODO; set K to the optimal value that you've found via experimentation
    const int MAX_ATTEMPTS = 10000;     //Maximum number of clustering attempts, you may want to use this
    const float threshold = 0.02f;      //Threshold for cluster similarity, you may want to use this and alter it if so

    
    //TODO; fix this function
    public Dictionary<Datapoint, List<Datapoint>> Cluster(Datapoint[] datapoints)
    {
        // Select K random centers to start
        List<Datapoint> centers = new List<Datapoint>();
        Dictionary<Datapoint, List<Datapoint>> clustersByCenters = new Dictionary<Datapoint, List<Datapoint>>();

        while (centers.Count < K)
        {
            // Generate a random index less than the size of the array.  循环，基于K值随机获取多个数据中心：
            // Generate a random index less than the size of the array.  循环，基于K值随机获取多个数据中心：
            int randomIndex = Random.Range(0, datapoints.Length); // 获取随机序数；
            Datapoint randomCenter = datapoints[randomIndex]; // 设定为随机中心；

            if (!centers.Contains(randomCenter)) // 若未记录该中心值，则将其加入中心值列表；
            {
                centers.Add(randomCenter);
            }
        }

        //Instantiate clusters by these centers  基于中心实例化集群：
        foreach (Datapoint center in centers) // 循环，对中心值列表中的每一个中心值，调用clustersByCenters创建空的中心化聚类；
        {
            clustersByCenters.Add(center, new List<Datapoint>()); // 构建由 中心值 和 含周围值列表=空 组成的字典簇；
        }

        
        //Map each datapoint to the closest center  将各周围值对应到最近中心：
        foreach (Datapoint point in datapoints) // 循环，对每一个数据值参数初始化为 近邻中心设定=空 和 最近距离=无穷；
        {
            Datapoint closestCenter = null;
            float minDistance = float.PositiveInfinity;

            foreach (Datapoint currentCenter in centers) // 循环，基于各个中心值进行距离运算，若新距离优于已有最近距离，替换目标中心；
            {
                float currentDistance = Distance(point, currentCenter);
                if (currentDistance < minDistance)
                {
                    closestCenter = currentCenter;
                    minDistance = currentDistance;
                }
            }
            clustersByCenters[closestCenter].Add(point); // 基于获取的最近中心值，添加其对应周边值到聚类数据结构；
        }
        
        
        // while not centroids==oldcentroids: 若原中心与新的中心差小于阀值，并且尝试次数未达上限：
        int attempts = 0;       // 记录尝试次数；
        List<Datapoint> oldCentroids = new List<Datapoint>();       // 创建旧质点列表，以用于进行质点更新；centers
        while ((DifferenceBetweenCenters(oldCentroids.ToArray(), centers.ToArray()) < threshold) && (attempts < MAX_ATTEMPTS))
        { 
            oldCentroids = new List<Datapoint>(centers.ToArray());    // 复制列表到新列表 centroids.ToArray().ToList();
            
            
            //Instantiate clusters by these centers  基于中心实例化集群：
            centers.Clear();
            foreach (List<Datapoint> pointsAround in clustersByCenters.Values)
            {
                Datapoint newCenter = GetAverage(pointsAround.ToArray()); 
                centers.Add(newCenter);
            }
            foreach (Datapoint center in centers) // 循环，对中心值列表中的每一个中心值，调用clustersByCenters创建空的中心化聚类；
            {
                if (clustersByCenters.ContainsKey(center))
                {
                    continue;
                }
                clustersByCenters.Add(center, new List<Datapoint>()); // 构建由 中心值 和 含周围值列表=空 组成的字典簇；
            }
            

            //Map each datapoint to its closest center  将各周围值对应到最近中心：
            foreach (Datapoint pnt in datapoints) // 循环，对每一个数据值参数初始化为 近邻中心设定=空 和 最近距离=无穷；
            {
                Datapoint closestCenter = null;
                float minDistance = float.PositiveInfinity;

                foreach (Datapoint center in centers) // 循环，基于各个中心值进行距离运算，若新距离优于已有最近距离，替换目标中心；
                {
                    float thisDistance = Distance(pnt, center);
                    if (thisDistance < minDistance)
                    {
                        closestCenter = center;
                        minDistance = thisDistance;
                    }
                }
                clustersByCenters[closestCenter].Add(pnt); // 基于获取的最近中心值，添加其对应周边值到聚类数据结构；
            }
        }
        return clustersByCenters;
    }


    //Calculate the difference between sets of centers
    private float DifferenceBetweenCenters(Datapoint[] centers1, Datapoint[] centers2)
    {
        List<Datapoint> mappedPoints = new List<Datapoint>();
        float totalDistance = 0;
        foreach(Datapoint c1 in centers1)
        {
            Datapoint minPoint = null;
            float minDistance = float.PositiveInfinity;

            foreach(Datapoint c2 in centers2)
            {
                if (!mappedPoints.Contains(c2))
                {
                    float thisDistance = Distance(c1, c2);

                    if (thisDistance < minDistance)
                    {
                        minDistance = thisDistance;
                        minPoint = c2;
                    }
                }
            }
            mappedPoints.Add(minPoint);
            totalDistance += minDistance;
        }
        
        return totalDistance;
    }

    //Calculate and returns the geometric median of the given datapoints
    public Datapoint GetMedian(Datapoint[] datapoints)
    {
        Datapoint medianPnt = null;
        float totalDistance = float.PositiveInfinity;

        for(int i = 0; i<datapoints.Length; i++)
        {
            float thisDistance = 0;
            for(int j=0; j<datapoints.Length; j++)
            {
                if (i != j)
                {
                    thisDistance += Distance(datapoints[i], datapoints[j]);
                }
            }

            if (thisDistance < totalDistance)
            {
                totalDistance = thisDistance;
                medianPnt = datapoints[i];
            }
        }

        return medianPnt;
    }

    //Calculate and returns the average of the given datapoints
    public Datapoint GetAverage(Datapoint[] datapoints)
    {
        Datapoint sumDatapoint = new Datapoint("", 0, 0, 0, 0, 0, 0, 0, 0);
        int churnedVal = 0;

        foreach(Datapoint d in datapoints)
        {
            sumDatapoint = new Datapoint("", sumDatapoint.HoursPlayed + d.HoursPlayed, sumDatapoint.Level + d.Level, sumDatapoint.PelletsEaten + d.PelletsEaten, sumDatapoint.FruitEaten + d.FruitEaten, sumDatapoint.GhostsEaten + d.GhostsEaten, sumDatapoint.AvgScore + d.AvgScore, sumDatapoint.MaxScore + d.MaxScore, sumDatapoint.TotalScore + d.TotalScore);
            
            if (d.Churned)
            {
                churnedVal += 1;
            }
            else
            {
                churnedVal -= 1;
            }
        }
        //Calculate averages
        int hoursPlayed = (int)(((float)sumDatapoint.HoursPlayed) / ((float)datapoints.Length));
        int level = (int)(((float)sumDatapoint.Level) / ((float)datapoints.Length));
        int pelletsEaten = (int)(((float)sumDatapoint.PelletsEaten) / ((float)datapoints.Length));
        int fruitEaten = (int)(((float)sumDatapoint.FruitEaten) / ((float)datapoints.Length));
        int ghostsEaten = (int)(((float)sumDatapoint.GhostsEaten) / ((float)datapoints.Length));
        float avgScore = (((float)sumDatapoint.AvgScore) / ((float)datapoints.Length));
        int maxScore = (int)(((float)sumDatapoint.MaxScore) / ((float)datapoints.Length));
        int totalScore = (int)(((float)sumDatapoint.MaxScore) / ((float)datapoints.Length));

        bool churned = false;
        if (churnedVal > 0)
        {
            churned = true;
        }

        return new Datapoint("", hoursPlayed, level, pelletsEaten, fruitEaten, ghostsEaten, avgScore, maxScore, totalScore, churned);
    }

    
    //TODO; alter this distance function
    //WARNING: DO NOT USE CHURNED AS PART OF THIS FUNCTION
    public static float Distance(Datapoint a, Datapoint b)
    {//HoursPlayed: 2143 Level: 334 Pellets Eaten: 509712 Fruit Eaten: 1319 Ghosts Eaten: 37927 Average Score: 2061 Max Score: 5010 Total Score: 359495
        float dist = 0;
        dist += (Mathf.Abs((1 / (1 + Mathf.Exp(-a.HoursPlayed))) - (1 / (1 + Mathf.Exp(-b.HoursPlayed)))));
        dist += (Mathf.Abs((1 / (1 + Mathf.Exp(-a.Level))) - (1 / (1 + Mathf.Exp(-b.Level)))));
        dist += (Mathf.Abs((1 / (1 + Mathf.Exp(-a.PelletsEaten))) - (1 / (1 + Mathf.Exp(-b.PelletsEaten)))));
        dist += (Mathf.Abs((1 / (1 + Mathf.Exp(-a.FruitEaten))) - (1 / (1 + Mathf.Exp(-b.FruitEaten)))));
        dist += (Mathf.Abs((1 / (1 + Mathf.Exp(-a.GhostsEaten))) - (1 / (1 + Mathf.Exp(-b.GhostsEaten)))));
        dist += (Mathf.Abs((1 / (1 + Mathf.Exp(-a.AvgScore))) - (1 / (1 + Mathf.Exp(-b.AvgScore)))));
        dist += (Mathf.Abs((1 / (1 + Mathf.Exp(-a.MaxScore))) - (1 / (1 + Mathf.Exp(-b.MaxScore)))));
        dist += (Mathf.Abs((1 / (1 + Mathf.Exp(-a.TotalScore))) - (1 / (1 + Mathf.Exp(-b.TotalScore)))));
        return dist;
    }

}

