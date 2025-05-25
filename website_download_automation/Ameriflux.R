# Make sure that you have these dependencies set up:

# !apt-get install -y r-base

# Install devtools and other required packages
# install.packages("devtools")
# install.packages("remotes")
# devtools::install_github("chuhousen/amerifluxr")

# Check if everything is installed and working

# library(amerifluxr)


# Comment out based on what you intend to do


# -------------------- Code -------------------- #
# Biogeophysical Information File (BIF) download

site <- amf_site_info()
floc1 <- amf_download_bif(user_id = "BrianX",
                          user_email = "brianxie1298@gmail.com",
                          data_policy = "CCBY4.0",
                          agree_policy = TRUE,
                          intended_use = "synthesis",
                          intended_use_text = "obtain AmeriFlux sites' geolocation, IGBP, and climate classification",
                          out_dir = tempdir(),
                          verbose = TRUE,
                          site_w_data = TRUE)

# -------------------- Code -------------------- #
# BASE-BADM Data Product for One Site

bif <- amf_read_bif(file = floc1)

floc2 <- amf_download_base(user_id = "BrianX",
                           user_email = "BrianXie1298@mail.com",
                           site_id = "US-CRT",
                           data_product = "BASE-BADM",
                           data_policy = "CCBY4.0",
                           agree_policy = TRUE,
                           intended_use = "remote_sensing",
                           intended_use_text = "validate the model of GPP estimation",
                           verbose = TRUE,
                           out_dir = tempdir())

# -------------------- Code -------------------- #
# BASE-BADM Data Product for Multiple Sites

base <- amf_read_base(file = floc2,
                      unzip = TRUE,
                      parse_timestamp = TRUE)

amf_download_base(user_id = "BrianX",
                  user_email = "BrianXie1298@mail.com",
                  site_id = c("US-CRT", "US-WPT", "US-Oho"),
                  data_product = "BASE-BADM",
                  data_policy = "CCBY4.0",
                  agree_policy = TRUE,
                  intended_use = "model",
                  intended_use_text = "Data-driven modeling, for training models and cross-validation",
                  verbose = TRUE,
                  out_dir = tempdir())
            
# -------------------- Code -------------------- #
