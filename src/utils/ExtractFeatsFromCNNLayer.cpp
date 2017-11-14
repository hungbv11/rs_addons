#include <uima/api.hpp>
#include <ros/package.h>

//#include <pcl/point_types.h>
//#include <pcl/filters/extract_indices.h>
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/impl/kdtree.hpp>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include <opencv/highgui.h>

#include <rs/recognition/CaffeProxy.h>

#include <dirent.h>
#include <fstream>

#include <yaml-cpp/yaml.h>

#include <boost/program_options.hpp>

#define TRAIN_DIR "/objects_dataset/partial_views/"

#define CAFFE_MODEL_FILE  "/caffe/models/bvlc_reference_caffenet/deploy.prototxt"
#define CAFFE_TRAINED_FILE  "/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
#define CAFFE_MEAN_FILE  "/caffe/data/imagenet_mean.binaryproto"
#define CAFFE_LABLE_FILE  "/caffe/data/synset_words.txt"

namespace po = boost::program_options;

enum FeatType
{
  VFH = 0,
  CVFH,
  CNN
};

void getFiles(const std::string &path,
              std::map<std::string, std::string> objectToLabel,
              std::map<std::string, std::vector<std::string>> &modelFiles,
              std::string fileExtension)
{
  DIR *classdp;
  struct dirent *classdirp;
  size_t pos;

  for(auto const & p : objectToLabel)
  {

    std::string pathToObj = path + p.first;
    classdp = opendir(pathToObj.c_str());
    if(classdp == NULL)
    {
      std::cerr << "<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      std::cerr << "FOLDER DOES NOT EXIST: " << pathToObj << std::endl;
      std::cerr << "<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      continue;
    }

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
        modelFiles[p.second].push_back(pathToObj + "/" + filename);
      }
    }
  }

  std::map<std::string, std::vector<std::string>>::iterator it;
  for(it = modelFiles.begin(); it != modelFiles.end(); ++it)
  {
    std::sort(it->second.begin(), it->second.end());
  }
}

void savetoFlann(const  std::vector<std::pair<std::string, std::vector<float> > > &features,
                 std::string featName, std::string splitName)
{
  if(features.size() > 0)
  {
    flann::Matrix<float> data(new float[features.size()*features[0].second.size()],
                              features.size(),
                              features[0].second.size());
    for(size_t i = 0; i < data.rows; ++i)
      for(size_t j = 0; j < data.cols; ++j)
      {
        data[i][j] = features[i].second[j];
      }
    std::string packagePath = ros::package::getPath("rs_addons");
    std::string savePath = packagePath +  "/data/extracted_feats/";
    flann::save_to_file(data, savePath + featName + "_" + splitName + ".hdf5", "training_data");
    std::ofstream fs;
    fs.open(savePath + featName + "_" + splitName + ".list");
    for(size_t i = 0; i < features.size(); ++i)
    {
      fs << features[i].first << "\n";
    }
    fs.close();
    flann::Index<flann::ChiSquareDistance<float> > index(data, flann::LinearIndexParams());
    index.buildIndex();
    index.save(savePath + featName + "_" + splitName + ".idx");
    std::cerr << "Saved data to : " << savePath << std::endl;
    delete[] data.ptr();
  }
}

void savetoYaml(const std::vector<std::pair<std::string, std::vector<float> > > &features,
                std::string featName, std::string splitName)
{

  if(!features.empty())
  {
    cv::FileStorage fs;
    std::string packagePath = ros::package::getPath("rs_addons");
    std::string savePath = packagePath +  "/data/extracted_feats/";
    fs.open(savePath + featName + "_" + splitName + ".yaml.gz", cv::FileStorage::WRITE);
    fs << "labels" << "[";
    cv::Mat descriptors(features.size(), features[0].second.size(), CV_32F);
    for(size_t i = 0; i < features.size(); ++i)
    {
      fs << features[i].first;
      for(size_t j = 0; j < features[i].second.size(); ++j)
      {
        descriptors.at<float>(i, j) = features[i].second[j];
      }
    }
    fs << "]";
    fs << "Descriptors" << descriptors;
    fs.release();
  }
}

void extractCNNFeature(const std::map<std::string, std::vector<std::string>> &modelFiles,
                       std::string resourcesPackagePath,
                       std::vector<std::pair<std::string, std::vector<float> > > &cnn_features)
{
  CaffeProxy caffeProxyObj(resourcesPackagePath + CAFFE_MODEL_FILE,
                           resourcesPackagePath + CAFFE_TRAINED_FILE,
                           resourcesPackagePath + CAFFE_MEAN_FILE,
                           resourcesPackagePath + CAFFE_LABLE_FILE);

  for(std::map<std::string, std::vector<std::string>>::const_iterator it = modelFiles.begin();
      it != modelFiles.end(); ++it)
  {
    std::cerr << it->first << std::endl;
    for(int i = 0; i < it->second.size(); ++i)
    {
      std::cerr << it->second[i] << std::endl;
      cv::Mat rgb = cv::imread(it->second[i]);
      std::vector<float> feature = caffeProxyObj.extractFeature(rgb);

      cv::Mat desc(1, feature.size(), CV_32F, &feature[0]);
      cv::normalize(desc, desc, 1, 0, cv::NORM_L2);
      std::vector<float> descNormed;
      descNormed.assign((float *)desc.datastart, (float *)desc.dataend);
      cnn_features.push_back(std::pair<std::string, std::vector<float>>(it->first, descNormed));
    }
  }
  std::cerr << "cnn_features size: " << cnn_features.size() << std::endl;
}

void extractVFHDescriptors(const std::map<std::string, std::vector<std::string>> &modelFiles,
                           std::vector<std::pair<std::string, std::vector<float> > > &vfh_features)
{
  //TODO: add preprocessing ?? e.g. smoothing, used to help
  pcl::VFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> vfhEstimation;
  for(std::map<std::string, std::vector<std::string>>::const_iterator it = modelFiles.begin();
      it != modelFiles.end(); ++it)
  {
    std::cerr << it->first << std::endl;
    for(int i = 0; i < it->second.size(); ++i)
    {
      std::cerr << it->second[i] << std::endl;
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
      pcl::io::loadPCDFile(it->second[i], *cloud);

      pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
      ne.setInputCloud(cloud);

      pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA> ());
      ne.setSearchMethod(tree);
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
      ne.setRadiusSearch(0.03);
      ne.compute(*cloud_normals);

      vfhEstimation.setInputCloud(cloud);
      vfhEstimation.setInputNormals(cloud_normals);
      vfhEstimation.setNormalizeBins(true);
      vfhEstimation.setNormalizeDistance(true);
      vfhEstimation.setSearchMethod(tree);

      pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs(new pcl::PointCloud<pcl::VFHSignature308> ());
      vfhEstimation.compute(*vfhs);
      std::vector<float> vfhsVec;
      vfhsVec.resize(308);
      for(size_t j = 0; j < 308; ++j)
      {
        vfhsVec[j] = vfhs->points[0].histogram[j];
      }
      vfh_features.push_back(std::pair<std::string, std::vector<float>>(it->first, vfhsVec));
    }
  }

  std::cerr << "vfh_features size: " << vfh_features.size() << std::endl;

}



int main(int argc, char **argv)
{

  po::options_description desc("Allowed options");
  std::string split, feat, saveTo;
  FeatType ft;
  desc.add_options()
  ("help,h", "Print help messages")
  ("split,s", po::value<std::string>(&split)->default_value("all"),
   "split file to use")
  ("feature,f", po::value<std::string>(&feat)->default_value("CNN"),
   "choose feature to extract: [VFH|CVFH|CNN]")
  ("saveTo,o", po::value<std::string>(&saveTo)->default_value("FLANN"),
   "choose output format : [YAML|FLANN]");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if(vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }

  if(feat == "VFH")
  {
    ft = VFH;
  }
  else if(feat == "CVFH")
  {
    ft = CVFH;
  }
  else if(feat == "CNN")
  {
    ft = CNN;
  }
  std::string packagePath = ros::package::getPath("rs_resources");

  std::string splitFilePath = split;
  std::string splitName;
  if(!boost::filesystem::exists(boost::filesystem::path(split)))
  {
    splitFilePath = packagePath + "/objects_dataset/splits/" + split + ".yaml";
    splitName = split;
  }
  else
  {
    splitName = split.substr(split.find_last_of("/")+1, split.find_last_of(".")-split.find_last_of("/")-1);
  }

  std::cout << "Path to split file: " << splitFilePath << std::endl;
  std::cout << "Name of split: " << splitName << std::endl;
  //label to file
  std::map<std::string, std::vector<std::string> > modelFilesPNG;
  std::map<std::string, std::vector<std::string> > modelFilesPCD;


  cv::FileStorage fs;
  fs.open(splitFilePath, cv::FileStorage::READ);
  std::vector<std::string> classes;


  std::map<std::string, std::string> objectToLabel;
  fs["classes"] >> classes;

  if(classes.empty())
  {
    std::cerr << "Split file has no classes defined" << std::endl;
    return false;
  }
  else
  {
    for(auto c : classes)
    {
      std::vector<std::string> subclasses;
      fs[c] >> subclasses;
      if(!subclasses.empty())
        for(auto sc : subclasses)
        {
          objectToLabel[sc] = c;
        }
      else
      {
        objectToLabel[c] = c;
      }
    }
  }

  getFiles(packagePath + TRAIN_DIR, objectToLabel, modelFilesPNG, "_crop.png");
  getFiles(packagePath + TRAIN_DIR, objectToLabel, modelFilesPCD, ".pcd");


  //Extract the feat descriptors
  std::vector<std::pair<std::string, std::vector<float> > > descriptors;
  switch(ft)
  {
  case CNN:
    extractCNNFeature(modelFilesPNG, packagePath, descriptors);

    break;
  case VFH:
    extractVFHDescriptors(modelFilesPCD, descriptors);
    break;
  default:
    std::cerr << "This is weird" << std::endl;
  }

  //save to disk
  if(saveTo == "FLANN")
  {
    std::cerr << "Saving as Flann" << std::endl;
    savetoFlann(descriptors, feat, splitName);
  }
  else if(saveTo == "YAML")
  {
    std::cerr << "Saving as Yaml" << std::endl;
    savetoYaml(descriptors, feat, splitName);
  }

  return true;
}
