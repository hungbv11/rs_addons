// Developed by: Rakib

#include<iostream>
#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <vector>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include <boost/filesystem.hpp>
#include<ros/package.h>
#include <rs/recognition/CaffeProxy.h>
#include <dirent.h>
#include <yaml-cpp/yaml.h>
#include <pcl/io/pcd_io.h>
#include<algorithm>
#include <iterator>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <boost/program_options.hpp>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>

using namespace cv;
using namespace std;
namespace po = boost::program_options;

// To read the descriptors matrix and it's label from /rs_resources/objects_dataset/extractedFeat folder...........
void readDescriptor(std::string matrix_name, std::string label_name,
                                          cv::Mat &des_matrix, cv::Mat &des_label)
{
  cv::FileStorage fs;
  std::string packagePath = ros::package::getPath("rs_resources") + '/';
  std::string savePath = "objects_dataset/extractedFeat/";

  if(!boost::filesystem::exists(packagePath + savePath+ matrix_name+".yaml")||
     !boost::filesystem::exists(packagePath + savePath+ label_name+".yaml"))
  {
    std::cout<<matrix_name <<" or "<<label_name <<" in path  ( " << packagePath + savePath << " ) does not exist. please check" << std::endl;
  }
  else
  {
    fs.open(packagePath + savePath + matrix_name + ".yaml", cv::FileStorage::READ);
    fs[matrix_name] >> des_matrix;

    fs.open(packagePath + savePath + label_name + ".yaml", cv::FileStorage::READ);
    fs [label_name] >> des_label;
  }
}

// To save the train and test data in cv::Mat format in folder /rs_resource/extractedFeat
void saveDatasets (cv::Mat descriptors_train, cv::Mat label_train, std::string descriptor_name,
                   std::string dataset_name,std::string inputStorage, std::string savePathToOutput )
{


  //To save file in disk...........................................................
  cv::FileStorage fs;

  // To save the train data.................................................
  fs.open(savePathToOutput + dataset_name +'_'+ descriptor_name +'_'+"MatTrain"+'_'+ inputStorage+".yaml", cv::FileStorage::WRITE);
  fs <<dataset_name +'_'+ descriptor_name +'_'+"MatTrain"+'_'+inputStorage << descriptors_train;
  fs.release();
  fs.open(savePathToOutput + dataset_name +'_'+ descriptor_name +'_'+"MatTrainLabel"+'_'+inputStorage+".yaml", cv::FileStorage::WRITE);
  fs <<dataset_name +'_'+ descriptor_name +'_'+"MatTrainLabel"+'_'+inputStorage<< label_train;
  fs.release();

  std::cout<<"Generated feautres should be found in path ("<< savePathToOutput<<")"<<std::endl;
}



int main(int argc, char **argv)

{

  po::options_description desc("Allowed options");
  std::string first_mat_name, first_label_name, second_mat_name, second_label_name;
  desc.add_options()
      ("help,h", "Print help messages")
      ("MatTrain1,m1", po::value<std::string>(&first_mat_name)->default_value("IAI_CNN_MatTrain_objectsNkit25badNkit20bad"),
       "enter First matrix name")
      ("MatTrainLabel1,l1", po::value<std::string>(&first_label_name)->default_value("IAI_CNN_MatTrainLabel_objectsNkit25badNkit20bad"),
       "enter enter First label name")
      ("MatTrain2,m2", po::value<std::string>(&second_mat_name)->default_value("DB_CNN_MatTrain_kit15bad"),
       "enter second matrix name")
      ("MatTrainLabel2,l2", po::value<std::string>(&second_label_name)->default_value("DB_CNN_MatTrainLabel_kit15bad"),
       "enter second label name");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if(vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }


  std::vector<std::string> sp_mat_first;
  std::vector<std::string> sp_mat_second;

  boost::split(sp_mat_first, first_mat_name, boost::is_any_of("_"));
  boost::split(sp_mat_second, second_mat_name, boost::is_any_of("_"));

  std::string dataset = sp_mat_first[0];
  std::string  feat= sp_mat_first[1];
  std::string combine_storage =sp_mat_first[3]+"N"+sp_mat_second[3];


  std::string packagePath = ros::package::getPath("rs_resources") + '/';
  std::string savePath = "objects_dataset/extractedFeat/";

  if(!boost::filesystem::exists(packagePath + savePath))
  {
    std::cout<< "path " << packagePath + savePath << " does not exist to save file. please check" << std::endl;
  }



   cv::Mat des_mat_first;
   cv::Mat des_lab_first;
 readDescriptor(first_mat_name, first_label_name, des_mat_first, des_lab_first);
   std::cout<<"first mat size:"<<des_mat_first.size()<<std::endl;
   std::cout<<"first label size:"<<des_lab_first.size()<<std::endl;

   cv::Mat des_mat_second;
   cv::Mat des_lab_second;
  readDescriptor(second_mat_name, second_label_name, des_mat_second, des_lab_second);
  std::cout<<"second mat size:"<<des_mat_second.size()<<std::endl;
  std::cout<<"second label size:"<<des_lab_second.size()<<std::endl;

 
  cv::Mat concat_mat;
  cv::Mat concat_lab;
  
  cv::vconcat(des_mat_first, des_mat_second, concat_mat);
  cv::vconcat(des_lab_first, des_lab_second, concat_lab);


  saveDatasets (concat_mat,concat_lab, feat,dataset, combine_storage, packagePath + savePath);
  std::cout<<dataset+"_"+feat+"_MatTrain_"+combine_storage<<std::endl;
  std::cout<<"size:"<<concat_mat.size()<<endl;
  std::cout<<dataset+"_"+feat+"_MatTrainLabel_"+combine_storage<<std::endl;
  std::cout<<"size:"<<concat_lab.size()<<endl;
  std::cout<<"Calculation is done"<<std::endl;

  return 0;
}  
