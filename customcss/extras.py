import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import random

def custom_btn(title,icon,key,on_click_handle=None):
    css = r"""
            button div:before {
                font-family: 'Font Awesome 5 Free';
                content: '"""
    css = css +  '\\'+icon
    css = css + r"""';
                display: inline-block;
                padding-right: 3px;
                vertical-align: middle;
                font-weight: 900;
            }
            """
    with stylable_container(
        key="container_with_border",
        css_styles=css
    ):
        st.button(title,key="btn-"+key,on_click=on_click_handle)
