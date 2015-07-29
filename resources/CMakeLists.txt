# resource files
set(RESOURCE_FILES 
	cleap_doxy_logo.png
	cleap_doxy_logo_white.png
	cleap_icon_bw.png
	cleap_icon_wb.png
	cleap_wb.png
	gplv3.png)

# install as shared files
INSTALL(FILES ${RESOURCE_FILES} DESTINATION share/${CLEAP_OUTPUT_DIR})
