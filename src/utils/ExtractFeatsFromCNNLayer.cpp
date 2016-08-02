#include <uima/api.hpp>
#include <ros/package.h>

//#include <pcl/point_types.h>
//#include <pcl/filters/extract_indices.h>
//#include <pcl_conversions/pcl_conversions.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/features/vfh.h>
//#include <pcl/features/normal_3d.h>
#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include <opencv/highgui.h>

#include <rs/recognition/CaffeProxy.h>

#include <dirent.h>
#include <fstream>

#include <yaml-cpp/yaml.h>

#include <boost/program_options.hpp>

#define TRAIN_DIR "/data/training_robohow"
//#define TRAIN_DIR "objects_dataset/partial_views"
#define CAFFE_DIR "/home/balintbe/local/src/caffe"
#define CAFFE_MODEL_FILE CAFFE_DIR "/models/bvlc_reference_caffenet/deploy.prototxt"
#define CAFFE_TRAINED_FILE CAFFE_DIR "/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
#define CAFFE_MEAN_FILE CAFFE_DIR "/data/ilsvrc12/imagenet_mean.binaryproto"
#define CAFFE_LABLE_FILE CAFFE_DIR "/data/ilsvrc12/synset_words.txt"

namespace po = boost::program_options;


class FileHandler
{
  FileHandler();
};

void getFiles(const std::string &path, std::map<std::string, std::vector<std::string>> &modelFiles, std::string fileExtension)
{
  DIR *dp;
  struct dirent *dirp;
  size_t pos;

  if((dp  = opendir(path.c_str())) ==  NULL)
  {
    std::cerr << "Error opening: " << std::endl;
    return;
  }

  while((dirp = readdir(dp)) != NULL)
  {
    std::string classname = dirp->d_name;
    if(strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0)
    {
      continue;
    }

    if(dirp->d_type == DT_DIR)
    {
      DIR *classdp;
      struct dirent *classdirp;
      std::string pathToClass = path + "/" + classname;
      classdp = opendir(pathToClass.c_str());
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
          modelFiles[classname].push_back(pathToClass + "/" + filename);
        }
      }
    }
  }
  closedir(dp);

  std::map<std::string, std::vector<std::string>>::iterator it;
  for(it = modelFiles.begin(); it != modelFiles.end(); ++it)
  {
    std::sort(it->second.begin(), it->second.end());
  }
}


void extractCNNFeature(const std::map<std::string, std::vector<std::string>> &modelFiles)
{
  CaffeProxy caffeProxyObj(CAFFE_MODEL_FILE,
                           CAFFE_TRAINED_FILE,
                           CAFFE_MEAN_FILE,
                           CAFFE_LABLE_FILE);

  std::vector<std::pair<std::string, std::vector<float> > > cnn_features;

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
      cv::normalize(desc,desc,1,0,cv::NORM_L2);
      std::vector<float> descNormed;
      descNormed.assign((float *)desc.datastart,(float*)desc.dataend);
      cnn_features.push_back(std::pair<std::string, std::vector<float>>(it->first, descNormed));
    }
  }
  std::cerr << "cnn_features size: " << cnn_features.size() << std::endl;
  if(cnn_features.size() > 0)
  {
    flann::Matrix<float> data(new float[cnn_features.size()*cnn_features[0].second.size()],
                              cnn_features.size(),
                              cnn_features[0].second.size());
    for(size_t i = 0; i < data.rows; ++i)
      for(size_t j = 0; j < data.cols; ++j)
      {
        data[i][j] = cnn_features[i].second[j];
      }
    std::string packagePath = ros::package::getPath("rs_addons");
    std::string savePath = packagePath + TRAIN_DIR;
    flann::save_to_file (data, savePath+"/cnnfc7.hdf5", "training_data");
    std::ofstream fs;
    fs.open (savePath+"/cnnfc7.list");
    for (size_t i = 0; i < cnn_features.size (); ++i)
      fs << cnn_features[i].first << "\n";
    fs.close ();
    flann::Index<flann::ChiSquareDistance<float> > index (data, flann::LinearIndexParams ());
    index.buildIndex ();
    index.save (savePath+"/kdtree.idx");
    delete[] data.ptr ();

  }

}




int main(int argc, char **argv)
{

//  po::options_description desc("Allowed options");
//  desc.add_options()
//      ("help", "produce help message")
//      ("compression", po::value<int>(), "set compression level")
//  ;

//  po::variables_map vm;
//  po::store(po::parse_command_line(argc, argv, desc), vm);
//  po::notify(vm);

//  if (vm.count("help")) {
//      std::cout << desc << "\n";
//      return 1;
//  }

//  if (vm.count("compression")) {
//      std::cout << "Compression level was set to "
//   << vm["compression"].as<int>() << ".\n";
//  } else {
//      std::cout << "Compression level was not set.\n";
//  }

  std::map<std::string, std::vector<std::string> > modelFilesPNG;

  std::string packagePath = ros::package::getPath("rs_addons");
  getFiles(packagePath + TRAIN_DIR, modelFilesPNG, "_crop.png");
  extractCNNFeature(modelFilesPNG);

  return true;
}
