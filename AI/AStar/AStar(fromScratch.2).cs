using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
///  Last Edition - 20211004 - Eden Zhou
/// </summary>
/// 

public class AStarPathFinder : GreedyPathFinder
{
    public static int nodesOpened = 0;
    
    //ASSIGNMENT 2: EDIT BELOW THIS LINE, IMPLEMENT A*
    public override Vector3[] CalculatePath(GraphNode startNode, GraphNode goalNode)
    {
        nodesOpened = 0;
        
        Dictionary<GraphNode, float> gScores = new Dictionary<GraphNode, float>();
        Dictionary<GraphNode, AStarNode> previousNodes = new Dictionary<GraphNode, AStarNode>();
        List<GraphNode> closedSet = new List<GraphNode>();
        
        PriorityQueue<AStarNode> openSet = new PriorityQueue<AStarNode>();
        AStarNode start = new AStarNode(null, startNode, Heuristic(startNode, goalNode));
        openSet.Enqueue(start);
        
        int attempts = 0;
        while (openSet.Count() > 0 && attempts < 10000) 
        {
            attempts += 1;
            AStarNode currNode = openSet.Dequeue();
            closedSet.Add(currNode.GraphNode);
            
            if (currNode.Location == goalNode.Location) 
            {
                Debug.Log("CHECKED " + nodesOpened + " NODES");
                return ReconstructPath(start, currNode);
            }
            
            foreach (GraphNode neighbor in currNode.GraphNode.Neighbors)
            {
                float gScore = currNode.GetGScore() + ObstacleHandler.Instance.GridSize;
                
                if (closedSet.Contains(neighbor))
                {
                    continue;  
                }
                if (previousNodes.ContainsKey(neighbor))        
                {
                    if (gScore < gScores[neighbor])
                    {
                        openSet.Remove(previousNodes[neighbor]);
                    }
                    else
                    {
                        continue;
                    }
                }
                AStarNode aStarNeighbor = new AStarNode(currNode, neighbor, Heuristic(neighbor, goalNode));
                openSet.Enqueue(aStarNeighbor);
                previousNodes[neighbor] = aStarNeighbor;
                gScores[neighbor] = gScore;
            }
        }
        Debug.Log("CHECKED " + nodesOpened + " NODES");
        return null;
    }
    //ASSIGNMENT 2: EDIT ABOVE THIS LINE, IMPLEMENT A*

    
    
    
    
    //EXTRA CREDIT ASSIGNMENT 2 EDIT BELOW THIS LINE
    public float Heuristic(GraphNode currNode, GraphNode goalNode)
    {
        float deltaX = currNode.Location.x - goalNode.Location.x;
        float deltaY = currNode.Location.y - goalNode.Location.y;
        
        if (deltaX > deltaY)
        {
            return 1000 * deltaX + 1 * deltaY;
        }
        else if (deltaX < deltaY)
        {
            return 1 * deltaX + 1000 * deltaY;
        }
        else
        {
            return (Mathf.Max(Mathf.Abs(currNode.Location.x - goalNode.Location.x), Mathf.Abs(currNode.Location.y - goalNode.Location.y)));
        }
    }
    
    //EXTRA CREDIT ASSIGNMENT 2 EDIT ABOVE THIS LINE

    
    
    
    
    //Code for reconstructing the path, don't edit this.
    private Vector3[] ReconstructPath(AStarNode startNode, AStarNode currNode)
    {
        List<Vector3> backwardsPath = new List<Vector3>();

        while (currNode != startNode)
        {
            backwardsPath.Add(currNode.Location);
            currNode = currNode.Parent;
        }
        backwardsPath.Reverse();

        return backwardsPath.ToArray();
    }
}



