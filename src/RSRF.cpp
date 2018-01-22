// Developed by: Rakib

#include<iostream>
#include <vector>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include <map>
#include <yaml-cpp/yaml.h>
#include<ros/package.h>
#include<boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <rs_addons/RSClassifier.h>
#include <rs_addons/RSRF.h>
using namespace cv;

//..............................Random Forest Classifier.........................................
RSRF::RSRF()
{

}

void RSRF:: trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name)
{
  cv::Mat train_matrix;
  cv::Mat train_label;
  readDescriptorAndLabel(train_matrix_name, train_label_name, train_matrix, train_label);
  std::cout << "size of train matrix:" << train_matrix.size() << std::endl;
  std::cout << "size of train label:" << train_label.size() << std::endl;

  std::string pathToSaveModel= saveTrained(trained_file_name);

  if(!pathToSaveModel.empty())
  {
    //set parameters for random forest algorithm ............................
    cv::Mat var_type = cv::Mat(train_matrix.cols + 1, 1, CV_8U);
    var_type.setTo(Scalar(CV_VAR_NUMERICAL));
    var_type.at<uchar>(train_matrix.cols, 0) = CV_VAR_CATEGORICAL;

    CvRTParams params = CvRTParams(25,    // max depth
                                   10,    // min sample count
                                   0,     // regression accuracy
                                   false, // compute surrogate split
                                   15,    // max number of categories
                                   NULL, // the array of priors
                                   false,  // calculate variable importance
                                   4,      // number of variables randomly selected at node and used to find the best split(s).
                                   100,    // max number of trees in the forest
                                   0.01f,  // forrest accuracy
                                   CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termination cirteria
                                   );


    CvRTrees *rtree = new CvRTrees;

    //train the random forest.....................................
    rtree->train(train_matrix, CV_ROW_SAMPLE, train_label, cv::Mat(), cv::Mat(), var_type, cv::Mat(), params);

    //To save the trained data.............................
    rtree->save((pathToSaveModel).c_str());
  }
}

int RSRF::predict_multi_class(cv::Mat sample, cv::AutoBuffer<int>& out_votes)
{

int result = 0;
int k;


if( nclasses > 0 ) //classification
{

    int* votes = out_votes;
    std::cout<<"number of trees:"<<ntrees<<std::endl;
    std::cout<<"number of n-classes:"<<nclasses<<std::endl;
    std::cout<<"size of memory:"<<sizeof(*votes)<<std::endl;
    memset( votes, 0, sizeof(*votes)*nclasses );
    for( k = 0; k < ntrees; k++ )
    {
        CvDTreeNode* predicted_node = trees[k]->predict( sample, cv::Mat());
         int nvotes;
         int class_idx = predicted_node->class_idx;
         //double val = predicted_node->value;
         //std::cout<<"class label:"<<val<<std::endl;
        CV_Assert( 0 <= class_idx && class_idx < nclasses );

       nvotes = ++votes[class_idx];

       std::cout<<"class index:"<< class_idx <<std::endl;

    }

    result = ntrees;
}

return result;

}

void RSRF:: classify(std::string trained_file_name_saved, std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble)
{
  cv::Mat test_matrix;
  cv::Mat test_label;
  readDescriptorAndLabel(test_matrix_name, test_label_name, test_matrix, test_label);
  std::cout << "size of test matrix :" << test_matrix.size() << std::endl;
  std::cout << "size of test label" << test_label.size() << std::endl;


  //To load the trained data................................
  load((loadTrained(trained_file_name_saved)).c_str());
  int numberOfCls=nclasses;
  int sizeOfTree=ntrees;
  std::cout<<numberOfCls<<sizeOfTree;

  //convert test label matrix into a vector.......................
  std::vector<double> con_test_label;
  test_label.col(0).copyTo(con_test_label);

  //Container to hold the integer value of labels............................
  std::vector<int> actual_label;
  std::vector<int> predicted_label;
   cv::AutoBuffer<int> out_votes;

  for(int i = 0; i < test_label.rows; i++)
  {
      double res = predict(test_matrix.row(i), cv::Mat());
      int prediction = res;

      int class_index=res-1;
      double numberOfTrees=predict_multi_class(test_matrix.row(i), out_votes);

      std::cout<<"number of trees: "<<numberOfTrees<<std::endl;
      std::cout<<"class : "<<res <<" gets : "<<out_votes[class_index]<<" votes"<<std::endl;
      std::cout<<"confidence of the RF Classifier: "<<out_votes[class_index]/numberOfTrees<<std::endl;

      predicted_label.push_back(prediction);
      double lab = con_test_label[i];
      int actual_convert = lab;
      actual_label.push_back(actual_convert);
    }
    std::cout << "Random forest Result :" << std::endl;
    evaluation(actual_label, predicted_label, obj_classInDouble);
}

void RSRF::classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det, double &confi)
{
    cv::AutoBuffer<int> out_votes;
     //To load the test data.............................
     std::cout << "size of test matrix :" << test_mat.size() << std::endl;

     //CvRTrees *vrtree = new CvRTrees;

     //To load the trained data................................
     load((loadTrained(trained_file_name_saved)).c_str());
     double res = predict(test_mat, cv::Mat());

     int class_index=res-1;
     double numberOfTrees=predict_multi_class(test_mat, out_votes);
     double con=out_votes[class_index]/numberOfTrees;
     std::cout<<"number of trees: "<<numberOfTrees<<std::endl;
     std::cout<<"class : "<<res <<" gets : "<<out_votes[class_index]<<" votes"<<std::endl;
     std::cout<<"confidence of the RF Classifier: "<<con<<std::endl;
     det = res;
     confi=con;
}

void RSRF::RsAnnotation(uima::CAS &tcas, std::string class_name, std::string feature_name, std::string database_name, rs::Cluster &cluster, std::string set_mode, double &confi)
{
    rs::ClassConfidence conResult = rs::create<rs::ClassConfidence>(tcas);
    conResult.score.set(confi);
    conResult.source("Random Forest");
    cluster.annotations.append(conResult);

    rs::Classification classResult = rs::create<rs::Classification>(tcas);
    classResult.classname.set(class_name);
    classResult.classifier("Random Forest");
    classResult.featurename(feature_name);
    classResult.model(database_name);

    if(set_mode == "CL")
    {
      outInfo("Setting annotation to cluster");
      cluster.annotations.append(classResult);
    }
    else if(set_mode == "GT")
    {
      rs::GroundTruth setGT = rs::create<rs::GroundTruth>(tcas);
      setGT.classificationGT.set(classResult);
      cluster.annotations.append(setGT);

    }
    else
    {
      outError("You should set the parameter (set_mode) as CL or GT"<<std::endl);
    }
}

RSRF::~ RSRF()
{
}
