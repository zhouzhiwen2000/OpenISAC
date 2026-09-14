if(NOT TARGET Boost::system)
    add_library(Boost::system INTERFACE IMPORTED)
    if(TARGET Boost::headers)
        set_property(TARGET Boost::system PROPERTY
            INTERFACE_LINK_LIBRARIES Boost::headers
        )
    endif()
endif()

set(boost_system_FOUND TRUE)
set(boost_system_VERSION "${OPENISAC_BOOST_VERSION}")
