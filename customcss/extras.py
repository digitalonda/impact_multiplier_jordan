import streamlit as st
from streamlit_extras.stylable_container import stylable_container

def custom_btn(title,icon):
    with stylable_container(
        key="container_with_border",
        css_styles=r"""
            button div:before {
                font-family: 'Font Awesome 5 Free';
                content: '\{icon}';
                display: inline-block;
                padding-right: 3px;
                vertical-align: middle;
                font-weight: 900;
            }
            """.format(icon=icon),
    ):
        st.button(title)
