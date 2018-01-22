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


// to get object to label from all_objects
double classLabelMapSimObjectsInDbl(std::string new_label, std::vector<std::string> all_objects)
{
   double a=0;

   for(int i=0; i<all_objects.size(); i++)
   {
     if(new_label==all_objects[i])
     {
       a=i+1;
     }
   }

   return a;
}


void getSplitFile(const std::string path,  std::vector<std::string> &class_name)
{
  std::ifstream file(path.c_str());
  std::string str;

  while(std::getline(file, str))
  {
   class_name.push_back(str);
  }
}

//To create object to class label map for new images

 double clsLabVector(string inputLabel, std::vector<std::string>vecLabel)
 {
   double clsDouble=0;
   for(int i=0; i<vecLabel.size();i++)
   {
     if(inputLabel==vecLabel[i])
     {
        clsDouble=i+1;
     }
   }

   return clsDouble;

 }



void getFilesDB(const std::string &path,
               std::vector<std::pair<double, std::string > > &modelFiles,
               std::vector <std::pair < string, double> > &classMapStToDbl,
               std::vector<std::string> all_objects, std::string fileExtension)
{
  DIR *classdp;
  struct dirent *classdirp;
  size_t pos;
  std::vector<std::string> classMapString;

   std::cout<<path<<std::endl;
  classdp = opendir(path.c_str());

  while((classdirp = readdir(classdp)) != NULL)
  {
    if(classdirp->d_type != DT_REG)
    {
      continue;
    }
    std::string filename = classdirp->d_name;
    pos = filename.rfind(fileExtension.c_str());
    if(pos != std::string::npos)
    {
      //To split the class name
      std::vector<std::string> cls_lab;
      boost::split(cls_lab, filename, boost::is_any_of("&"));

      double clsLabInDbl=classLabelMapSimObjectsInDbl(cls_lab[0], all_objects);

     //To check new label matches with label in all_objects
     if(clsLabInDbl == 0)
     {
       std::cerr << "new image label doesn't match with label in objects. Please have a look " << std::endl;
       break;
     }

     double clsFromVector= clsLabVector(cls_lab[0], classMapString);
     //To create object to class label for new images
     if(clsFromVector == 0)
     {
       classMapString.push_back(cls_lab[0]);
       classMapStToDbl.push_back(std::pair < string, double> (cls_lab[0],clsLabInDbl));

     }

     modelFiles.push_back(std::pair<double, std::string >(clsLabInDbl, path + "/" + filename));
    }
  }



  std::cout<<"classLabelMap:"<<endl;
  for(auto const & e : classMapStToDbl)
   {
      std::cout<<e.first<<"::"<<e.second<<std::endl;
    }

  std::cout<<"modelfiles:"<<endl;
  for(auto const & y : modelFiles)
   {
      std::cout<<y.first<<"::"<<y.second<<std::endl;
    }

}


//To get files from database image folder and from partial views............ 



// To extract the VFH feature.........................................
// Taken from rs_addons...............................................
void extractPCLDescriptors(std::string descriptorType, const std::vector<std::pair<double, std::string > > &modelFiles,
                           std::vector<std::pair<double, std::vector<float> > > &extract_features)
{

  std::string featDescription;

  for(auto i: modelFiles)
  {
    std::cerr << i.first<<"::" << i.second<< std::endl;

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
      pcl::io::loadPCDFile(i.second, *cloud);

      pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
      ne.setInputCloud(cloud);

      pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA> ());
      ne.setSearchMethod(tree);
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
      ne.setRadiusSearch(0.03);
      ne.compute(*cloud_normals);

      pcl::PointCloud<pcl::VFHSignature308>::Ptr extractedDiscriptor(new pcl::PointCloud<pcl::VFHSignature308> ());



      if (descriptorType=="VFH")
      {


        std::cout<<"Calculation start with VFH Feature"<<std::endl;

        pcl::VFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> vfhEstimation;
        vfhEstimation.setInputCloud(cloud);
        vfhEstimation.setInputNormals(cloud_normals);
        vfhEstimation.setNormalizeBins(true);
        vfhEstimation.setNormalizeDistance(true);
        vfhEstimation.setSearchMethod(tree);
        vfhEstimation.compute(*extractedDiscriptor);

        featDescription="VFH feature size :";

      }



      if(descriptorType=="CVFH")
      {

        std::cout<<"Calculation start with CVFH Feature"<<std::endl;

        pcl::CVFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> cvfhEst;
        cvfhEst.setInputCloud(cloud);
        cvfhEst.setInputNormals(cloud_normals);
        cvfhEst.setSearchMethod(tree);
        cvfhEst.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
        cvfhEst.setCurvatureThreshold(1.0);
        cvfhEst.setNormalizeBins(true);
        cvfhEst.compute(*extractedDiscriptor);

        featDescription="CVFH feature size :";
      }

      std::vector<float> descriptorVec;
      descriptorVec.resize(308);
      for(size_t j = 0; j < 308; ++j)
      {
        descriptorVec[j] = extractedDiscriptor->points[0].histogram[j];
      }
      extract_features.push_back(std::pair<double, std::vector<float> >(i.first, descriptorVec));

  }

  std::cerr << featDescription << extract_features.size() << std::endl;
}

// To extract the CNN and VGG16 features.........................................
void extractCaffeFeature(std::string featType, const  std::vector<std::pair<double, std::string > > &modelFiles,
                         std::string resourcesPackagePath,
                         std::vector<std::pair<double, std::vector<float> > > &caffe_features)
{
  std::string CAFFE_MODEL_FILE;
  std::string CAFFE_TRAINED_FILE;
  std::string featDescription;

  if(featType=="VGG16")
  {
    CAFFE_MODEL_FILE = "/caffe/models/bvlc_reference_caffenet/VGG_ILSVRC_16_layers_deploy.prototxt";
    CAFFE_TRAINED_FILE= "/caffe/models/bvlc_reference_caffenet/VGG_ILSVRC_16_layers.caffemodel";
    featDescription="VGG16 feature size :";
  }
  else if(featType=="CNN")
  {
    CAFFE_MODEL_FILE = "/caffe/models/bvlc_reference_caffenet/deploy.prototxt";
    CAFFE_TRAINED_FILE= "/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    featDescription="CNN feature size :";
  }
  else
  {
    std::cerr<<"CAFFE_MODEL_FILE and CAFFE_TRAINED_FILE are not found"<<std::cout;
    exit(0);
  }

  std::string CAFFE_MEAN_FILE= "/caffe/data/imagenet_mean.binaryproto";
  std::string CAFFE_LABLE_FILE = "/caffe/data/synset_words.txt";


  CaffeProxy caffeProxyObj(resourcesPackagePath + CAFFE_MODEL_FILE,
                           resourcesPackagePath + CAFFE_TRAINED_FILE,
                           resourcesPackagePath + CAFFE_MEAN_FILE,
                           resourcesPackagePath + CAFFE_LABLE_FILE);

  for(auto i: modelFiles)
  {
    std::cerr << i.first<<"::"<<i.second<< std::endl;

      cv::Mat rgb = cv::imread(i.second);
      std::vector<float> feature = caffeProxyObj.extractFeature(rgb);

      cv::Mat desc(1, feature.size(), CV_32F, &feature[0]);
      cv::normalize(desc, desc, 1, 0, cv::NORM_L2);
      std::vector<float> descNormed;
      descNormed.assign((float *)desc.datastart, (float *)desc.dataend);
      caffe_features.push_back(std::pair<double, std::vector<float>>(i.first, descNormed));

  }
  std::cerr << featDescription << caffe_features.size() << std::endl;
}


// To save the train and test data in cv::Mat format in folder /rs_resource/extractedFeat
void saveDatasets (std::vector<std::pair<double, std::vector<float> > > train_dataset,
                  std::string descriptor_name, std::string dataset_name, std::string inputStorage,std::string savePathToOutput )
{
  cv::Mat descriptors_train (train_dataset.size(), train_dataset[0].second.size(), CV_32F);
  cv::Mat label_train (train_dataset.size(), 1, CV_32F);

  for(size_t i = 0; i <  train_dataset.size(); ++i)
  {
    label_train.at<float>(i,0)= train_dataset[i].first;

    for(size_t j = 0; j <  train_dataset[i].second.size(); ++j)
    {
      descriptors_train.at<float>(i, j) =  train_dataset[i].second[j];
    }
  }

  //To save file in disk...........................................................
  cv::FileStorage fs;

  // To save the train data.................................................
  fs.open(savePathToOutput + dataset_name +'_'+ descriptor_name +'_'+"MatTrain"+'_'+ inputStorage+".yaml", cv::FileStorage::WRITE);
  fs <<dataset_name +'_'+ descriptor_name +'_'+"MatTrain"+'_'+inputStorage << descriptors_train;
  fs.release();
  fs.open(savePathToOutput + dataset_name +'_'+ descriptor_name +'_'+"MatTrainLabel"+'_'+inputStorage+".yaml", cv::FileStorage::WRITE);
  fs <<dataset_name +'_'+ descriptor_name +'_'+"MatTrainLabel"+'_'+inputStorage<< label_train;
  fs.release();

  std::cout<<"extracted feautres should be found in path ("<< savePathToOutput<<")"<<std::endl;
}


void saveObjectToLabels( std::vector <std::pair < string, double> > input_file,std::string descriptor_name,
                         std::string dataset_name, std::string inputStorage, std::string savePathToOutput)
{
  std::ofstream file((savePathToOutput + dataset_name +'_'+ descriptor_name +'_'+"ClassLabel"+'_'+inputStorage+".txt").c_str());

  for(auto p: input_file)
  {
    file<<p.first<<":"<<p.second<<endl;
  }

  std::cout<<" clasLabel in double should be found in path ("<< savePathToOutput<<")"<<std::endl;
}



int main(int argc, char **argv)

{

  po::options_description desc("Allowed options");
  std::string split_name,storageInput, feat,dataset_name;
  desc.add_options()
      ("help,h", "Print help messages")
      ("split,s", po::value<std::string>(& split_name)->default_value("objects"),
       "enter the split file name")
      ("storageInput,i", po::value<std::string>(& storageInput)->default_value("kit15bad"),
       "enter input storage folder name")
      ("datasets,d", po::value<std::string>(&dataset_name)->default_value("DB"),
       "choose the dataset: [DB]")
      ("feature,f", po::value<std::string>(&feat)->default_value("CNN"),
       "choose feature to extract: [CNN|VGG16|VFH|CVFH]");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if(vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }

  // Define path to get the splitfile.......................................................
  std::string resourcePath = ros::package::getPath("rs_resources");

  std::string object_file_path ;

  object_file_path = resourcePath + "/objects_dataset/splits/" + split_name + ".txt";



  if(!boost::filesystem::exists(boost::filesystem::path( object_file_path)))
  {
    std::cout<<"*********************************************************************************************"<<std::endl;
    std::cerr<<" Class label file (.yaml) is not found. Please check the path below  :"<<std::endl;
    std::cerr << "Path to class label file : " << object_file_path << std::endl<<std::endl;
    std::cerr << "The file should be in ( rs_resources/objects_datasets/splits/ ) folder "<< std::endl<<std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Path to class label file : " << object_file_path << std::endl<<std::endl;


  // Define path to get the datasets.......................................................
  std::string db_image= resourcePath +"/objects_dataset/"+ storageInput;



  //To save file in disk...........................................................

  std::string savePathToOutput = resourcePath +"/objects_dataset/extractedFeat/";

  // To check the storage folder for generated files by this program ................................................
  if(!boost::filesystem::exists(savePathToOutput))
  {
    std::cerr<<"Folder called (extractedFeat) not found to save the extracted feature generated by this code."<<std::endl;
    std::cerr<<"Please create the folder in rs_resource/objects_dataset/ and name it as extractedFeat "<<std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Path to save the extracted feature : " << savePathToOutput << std::endl<< std::endl;


   std::vector<std::string> all_object;
   getSplitFile(object_file_path,  all_object);

   for(auto p:all_object)
   {
     std::cout<<p<<std::endl;
   }

  std::vector<std::pair<double, std::string > > model_files_all;
   std::vector <std::pair < string, double> > objectToClassLabelMap;
  //getFilesDB(db_image, modelfil, ObjectToClassMap, "_.png");
  
  //Extract the feat descriptors
   std::vector<std::pair<double, std::vector<float> > > descriptors_all;


    getFilesDB(db_image, model_files_all, objectToClassLabelMap, all_object, "_.png");




    if(feat == "CNN")
    {
      std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

      // To calculate VFH descriptors..................................
      extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);
    }
    else if(feat == "VGG16")
    {
      std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

      // To calculate VFH descriptors..................................
      extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);
    }
    else if(feat == "VFH")
    {
      std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

      // To calculate VFH descriptors..................................
      extractPCLDescriptors(feat,model_files_all, descriptors_all);
    }
    else if(feat == "CVFH")
    {
      std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

      // To calculate VFH descriptors..................................
      extractPCLDescriptors(feat,model_files_all, descriptors_all);
    }
    else
    {
      std::cerr<<"Please select feature (CNN , VGG16, VFH, CVFH)"<<std::endl;
      return EXIT_FAILURE;
    }

    // To save the train and test data in path /rs_resources/objects_dataset/extractedFeat
    saveDatasets (descriptors_all, feat,dataset_name,storageInput,savePathToOutput);

    //To save the string class labels in type double in folder rs_resources/objects_dataset/extractedFeat
    saveObjectToLabels(objectToClassLabelMap,feat,dataset_name, storageInput,savePathToOutput);
  


  std::cout<<"Descriptors calculation is done"<<std::endl;

  return 0;
}  
