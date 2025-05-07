document.addEventListener('DOMContentLoaded', function() {
    // Toggle between signup and login
    const signUpButton = document.getElementById('signup');
    const signInButton = document.getElementById('login');
    const container = document.querySelector('.auth-container');

    signUpButton.addEventListener('click', () => {
        container.classList.add('right-panel-active');
    });

    signInButton.addEventListener('click', () => {
        container.classList.remove('right-panel-active');
    });

    // Form field focus and blur effects
    const formInputs = document.querySelectorAll('.form-group input');
    
    formInputs.forEach(input => {
        // Add 'focus' class on focus
        input.addEventListener('focus', () => {
            input.parentElement.classList.add('focus');
        });
        
        // Remove 'focus' class on blur
        input.addEventListener('blur', () => {
            input.parentElement.classList.remove('focus');
            
            // Add 'has-value' class if input has value
            if (input.value.trim() !== '') {
                input.parentElement.classList.add('has-value');
            } else {
                input.parentElement.classList.remove('has-value');
            }
        });
        
        // Check if input already has value on page load
        if (input.value.trim() !== '') {
            input.parentElement.classList.add('has-value');
        }
    });

    // Login form validation
    const loginForm = document.getElementById('loginForm');
    const loginEmail = document.getElementById('loginEmail');
    const loginPassword = document.getElementById('loginPassword');
    const loginEmailGroup = document.getElementById('loginEmailGroup');
    const loginPasswordGroup = document.getElementById('loginPasswordGroup');
    const loginEmailError = document.getElementById('loginEmailError');
    const loginPasswordError = document.getElementById('loginPasswordError');
    const loginButton = document.getElementById('loginButton');
    const loginSuccess = document.getElementById('loginSuccess');
    
    loginForm.addEventListener('submit', function(event) {
        event.preventDefault();
        let isValid = true;
        
        // Reset previous error states
        loginEmailGroup.classList.remove('error');
        loginPasswordGroup.classList.remove('error');
        
        // Validate email
        if (loginEmail.value.trim() === '') {
            showError(loginEmailGroup, loginEmailError, 'Email is required');
            isValid = false;
        } else if (!isValidEmail(loginEmail.value)) {
            showError(loginEmailGroup, loginEmailError, 'Please enter a valid email address');
            isValid = false;
        }
        
        // Validate password
        if (loginPassword.value.trim() === '') {
            showError(loginPasswordGroup, loginPasswordError, 'Password is required');
            isValid = false;
        } else if (loginPassword.value.length < 6) {
            showError(loginPasswordGroup, loginPasswordError, 'Password must be at least 6 characters');
            isValid = false;
        }
        
        if (isValid) {
            // Show loading state
            loginButton.classList.add('loading');
            
            // Simulate server request (remove in production)
            setTimeout(() => {
                loginSuccess.classList.add('show');
                loginButton.classList.remove('loading');
                
                // Actually submit the form
                setTimeout(() => {
                    loginForm.submit();
                }, 1000);
            }, 1500);
        }
    });
    
    // Signup form validation
    const signupForm = document.getElementById('signupForm');
    const signupName = document.getElementById('signupName');
    const signupEmail = document.getElementById('signupEmail');
    const signupPassword = document.getElementById('signupPassword');
    const signupNameGroup = document.getElementById('signupNameGroup');
    const signupEmailGroup = document.getElementById('signupEmailGroup');
    const signupPasswordGroup = document.getElementById('signupPasswordGroup');
    const signupNameError = document.getElementById('signupNameError');
    const signupEmailError = document.getElementById('signupEmailError');
    const signupPasswordError = document.getElementById('signupPasswordError');
    const signupButton = document.getElementById('signupButton');
    const signupSuccess = document.getElementById('signupSuccess');
    
    signupForm.addEventListener('submit', function(event) {
        event.preventDefault();
        let isValid = true;
        
        // Reset previous error states
        signupNameGroup.classList.remove('error');
        signupEmailGroup.classList.remove('error');
        signupPasswordGroup.classList.remove('error');
        
        // Validate name
        if (signupName.value.trim() === '') {
            showError(signupNameGroup, signupNameError, 'Name is required');
            isValid = false;
        } else if (signupName.value.length < 3) {
            showError(signupNameGroup, signupNameError, 'Name must be at least 3 characters');
            isValid = false;
        }
        
        // Validate email
        if (signupEmail.value.trim() === '') {
            showError(signupEmailGroup, signupEmailError, 'Email is required');
            isValid = false;
        } else if (!isValidEmail(signupEmail.value)) {
            showError(signupEmailGroup, signupEmailError, 'Please enter a valid email address');
            isValid = false;
        }
        
        // Validate password
        if (signupPassword.value.trim() === '') {
            showError(signupPasswordGroup, signupPasswordError, 'Password is required');
            isValid = false;
        } else if (signupPassword.value.length < 6) {
            showError(signupPasswordGroup, signupPasswordError, 'Password must be at least 6 characters');
            isValid = false;
        }
        
        if (isValid) {
            // Show loading state
            signupButton.classList.add('loading');
            
            // Simulate server request (remove in production)
            setTimeout(() => {
                signupSuccess.classList.add('show');
                signupButton.classList.remove('loading');
                
                // Actually submit the form
                setTimeout(() => {
                    signupForm.submit();
                }, 1000);
            }, 1500);
        }
    });
    
    // Show error message
    function showError(group, errorElement, message) {
        group.classList.add('error');
        errorElement.textContent = message;
    }
    
    // Email validation function
    function isValidEmail(email) {
        const re = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
        return re.test(String(email).toLowerCase());
    }
    
    // Live validation as user types
    loginEmail.addEventListener('input', function() {
        if (loginEmailGroup.classList.contains('error')) {
            if (loginEmail.value.trim() !== '' && isValidEmail(loginEmail.value)) {
                loginEmailGroup.classList.remove('error');
            }
        }
    });
    
    loginPassword.addEventListener('input', function() {
        if (loginPasswordGroup.classList.contains('error')) {
            if (loginPassword.value.trim() !== '' && loginPassword.value.length >= 6) {
                loginPasswordGroup.classList.remove('error');
            }
        }
    });
    
    signupName.addEventListener('input', function() {
        if (signupNameGroup.classList.contains('error')) {
            if (signupName.value.trim() !== '' && signupName.value.length >= 3) {
                signupNameGroup.classList.remove('error');
            }
        }
    });
    
    signupEmail.addEventListener('input', function() {
        if (signupEmailGroup.classList.contains('error')) {
            if (signupEmail.value.trim() !== '' && isValidEmail(signupEmail.value)) {
                signupEmailGroup.classList.remove('error');
            }
        }
    });
    
    signupPassword.addEventListener('input', function() {
        if (signupPasswordGroup.classList.contains('error')) {
            if (signupPassword.value.trim() !== '' && signupPassword.value.length >= 6) {
                signupPasswordGroup.classList.remove('error');
            }
        }
    });
});