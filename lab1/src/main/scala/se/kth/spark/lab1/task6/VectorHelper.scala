package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, DenseVector, Vectors}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    var result: Double = 0.0
    
    for(i <- 0 to (v1.size - 1)) result += v1.apply(i) * v2.apply(i)
    
    return result
  }

  def dot(v: Vector, s: Double): Vector = {
    var v2a = v.toArray
    return Vectors.dense(v2a.map(_*s))
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    var arr: Array[Double] = new Array(v1.size)
    
    for(i <- 0 to (v1.size - 1)) arr(i) = v1.apply(i) + v2.apply(i)
    
    return Vectors.dense(arr)
  }

  def fill(size: Int, fillVal: Double): Vector = {
    var arr: Array[Double] = new Array(size)
    
    return Vectors.dense(arr.map(_ =>fillVal))
  }
  
}