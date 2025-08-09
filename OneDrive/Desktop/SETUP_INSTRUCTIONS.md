# Lyly's Restaurant Website Improvements - Setup Instructions

## Overview
This document provides complete setup instructions for the enhanced Lyly's Restaurant website with online ordering and MonCash payment integration.

## 🚀 New Features Added

### ✅ Completed Improvements
1. **Online Ordering System**
   - Shopping cart functionality
   - Real-time cart updates
   - Quantity management
   - Order customization

2. **MonCash Payment Integration**
   - Full MonCash API integration
   - Secure payment processing
   - Order verification system
   - Fallback to cash payment

3. **Enhanced User Experience**
   - Mobile-responsive design
   - Accessibility features (ARIA labels, keyboard navigation)
   - Dark mode support
   - High contrast mode support
   - Reduced motion preferences

4. **Professional UI/UX**
   - Floating cart button with live counter
   - Sliding cart sidebar
   - Checkout form with validation
   - Success/error notifications
   - Order confirmation system

## 📁 Files Structure

```
lyly-restaurant/
├── lylys_menu.html          # Main website file (enhanced)
├── moncash_integration.js   # MonCash frontend integration
├── moncash_backend.php      # MonCash backend processing
├── orders.json              # Orders database (auto-created)
├── SETUP_INSTRUCTIONS.md    # This file
└── README.md               # Project documentation
```

## 🛠️ Setup Instructions

### Step 1: Basic Setup

1. **Upload Files**
   ```bash
   # Upload these files to your web server:
   - lylys_menu.html
   - moncash_integration.js
   - moncash_backend.php
   ```

2. **Test Basic Functionality**
   - Open `lylys_menu.html` in a browser
   - Test cart functionality (add/remove items)
   - Test cash payment flow (should open WhatsApp)

### Step 2: MonCash Integration Setup

#### A. Get MonCash Credentials

1. **Create MonCash Business Account**
   - Visit: https://moncashdfs.com/business
   - Create a business account
   - Complete verification process

2. **Access Sandbox Environment**
   - Login to MonCash business portal
   - Navigate to Developer/API section
   - Get your credentials:
     - Client ID
     - Client Secret
     - Business Key

3. **Test in Sandbox**
   - Use sandbox environment first
   - Test payment flows thoroughly
   - Verify webhook callbacks work

#### B. Configure Backend

1. **Update PHP Configuration**
   ```php
   // In moncash_backend.php, update:
   $moncashConfig = [
       'clientId' => 'YOUR_ACTUAL_CLIENT_ID',
       'clientSecret' => 'YOUR_ACTUAL_CLIENT_SECRET',
       'businessKey' => 'YOUR_ACTUAL_BUSINESS_KEY',
       'sandbox' => true // Set to false for production
   ];
   ```

2. **Environment Variables (Recommended)**
   ```bash
   # Create .env file or set server environment variables:
   MONCASH_CLIENT_ID=your_client_id
   MONCASH_CLIENT_SECRET=your_client_secret
   MONCASH_BUSINESS_KEY=your_business_key
   ```

3. **Set Permissions**
   ```bash
   # Ensure PHP can write to orders.json
   chmod 664 orders.json
   chmod 755 moncash_backend.php
   ```

### Step 3: Server Configuration

#### A. PHP Requirements

1. **Required PHP Extensions**
   ```bash
   # Ensure these are enabled:
   - curl
   - json
   - file_get_contents
   ```

2. **PHP Version**
   - Minimum: PHP 7.4+
   - Recommended: PHP 8.0+

#### B. Web Server Setup

1. **Apache Configuration**
   ```apache
   # Add to .htaccess:
   RewriteEngine On
   
   # Enable CORS for API calls
   Header always set Access-Control-Allow-Origin \"*\"
   Header always set Access-Control-Allow-Methods \"GET, POST, OPTIONS\"
   Header always set Access-Control-Allow-Headers \"Content-Type\"
   ```

2. **Nginx Configuration**
   ```nginx
   # Add to nginx.conf:
   location ~ \\.php$ {
       fastcgi_pass unix:/var/run/php/php8.0-fpm.sock;
       fastcgi_index index.php;
       include fastcgi_params;
   }
   
   # Enable CORS
   add_header Access-Control-Allow-Origin *;
   add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
   add_header Access-Control-Allow-Headers 'Content-Type';
   ```

### Step 4: Testing & Validation

#### A. Frontend Testing

1. **Cart Functionality**
   ```javascript
   // Test these features:
   - Add items to cart ✓
   - Remove items from cart ✓
   - Update quantities ✓
   - Calculate totals correctly ✓
   - Persist cart during session ✓
   ```

2. **Payment Flow Testing**
   ```javascript
   // Test both payment methods:
   - Cash payment (WhatsApp integration) ✓
   - MonCash payment (redirect flow) ✓
   - Error handling ✓
   - Success confirmations ✓
   ```

#### B. Backend Testing

1. **API Endpoints**
   ```bash
   # Test all endpoints:
   curl -X POST http://yoursite.com/moncash_backend.php?action=create_payment
   curl -X POST http://yoursite.com/moncash_backend.php?action=verify_payment
   curl -X POST http://yoursite.com/moncash_backend.php?action=process_cash_order
   ```

2. **MonCash Integration**
   - Test payment creation
   - Test payment verification
   - Test webhook handling
   - Verify order storage

### Step 5: Production Deployment

#### A. Security Checklist

1. **Environment Variables**
   ```bash
   # Never commit credentials to code
   # Use environment variables:
   export MONCASH_CLIENT_ID=your_production_client_id
   export MONCASH_CLIENT_SECRET=your_production_client_secret
   export MONCASH_BUSINESS_KEY=your_production_business_key
   ```

2. **HTTPS Setup**
   ```bash
   # MonCash requires HTTPS in production
   # Ensure SSL certificate is installed
   # Force HTTPS redirects
   ```

3. **File Permissions**
   ```bash
   chmod 644 lylys_menu.html
   chmod 644 moncash_integration.js
   chmod 755 moncash_backend.php
   chmod 664 orders.json
   ```

#### B. Go Live Steps

1. **Switch to Production**
   ```php
   // In moncash_backend.php:
   'sandbox' => false // Change to false
   ```

2. **Update URLs**
   - Ensure all URLs point to production domain
   - Update callback URLs in MonCash dashboard
   - Test payment flows in production

3. **Monitor & Maintain**
   - Set up error logging
   - Monitor order.json file size
   - Regular backup procedures

## 🔧 Customization Options

### A. Styling Customization

1. **Colors & Branding**
   ```css
   :root {
     --primary-color: #007bff; /* Change to your brand color */
     --text-color: #007bff;
     --bg-color: #ffffff;
   }
   ```

2. **Mobile Responsiveness**
   ```css
   /* Already implemented for all screen sizes */
   /* Customizable breakpoints in CSS */
   ```

### B. Functionality Extensions

1. **Additional Payment Methods**
   - Integrate other payment processors
   - Add cryptocurrency payments
   - Implement loyalty programs

2. **Enhanced Features**
   - Add customer accounts
   - Implement order tracking
   - Add reviews and ratings

## 📱 Mobile App Integration

The website is fully responsive and works as a Progressive Web App (PWA). To enhance mobile experience:

1. **Add to Home Screen**
   - Users can add website to home screen
   - Functions like a native app

2. **Push Notifications**
   - Can be added for order updates
   - Promotional notifications

## 🎯 Performance Optimization

### A. Loading Speed
- Optimized CSS and JavaScript
- Compressed images
- Minified code for production

### B. Caching Strategy
```apache
# Add to .htaccess for caching:
<IfModule mod_expires.c>
    ExpiresActive On
    ExpiresByType text/css \"access plus 1 month\"
    ExpiresByType application/javascript \"access plus 1 month\"
    ExpiresByType image/png \"access plus 1 year\"
</IfModule>
```

## 🛡️ Security Best Practices

1. **API Security**
   - Input validation on all forms
   - SQL injection prevention (using JSON, not SQL)
   - XSS protection through proper encoding

2. **MonCash Security**
   - Secure credential storage
   - HTTPS-only in production
   - Webhook signature verification

## 📞 Support & Maintenance

### A. Regular Tasks
- Monitor orders.json file size
- Check payment success rates
- Update MonCash credentials when expired
- Regular security updates

### B. Troubleshooting
- Check error logs for PHP errors
- Verify MonCash API status
- Test WhatsApp integration
- Validate form submissions

## 🔍 Analytics & Tracking

Consider adding:
- Google Analytics for visitor tracking
- Order completion rates
- Payment method preferences
- Mobile vs desktop usage

## 📈 Future Enhancements

Potential additions:
- Customer reviews system
- Inventory management
- Delivery tracking
- Multi-language support
- Advanced reporting dashboard

---

## Quick Start Checklist

- [ ] Upload all files to web server
- [ ] Test basic cart functionality
- [ ] Get MonCash credentials
- [ ] Configure moncash_backend.php
- [ ] Test payment flows
- [ ] Set up HTTPS
- [ ] Switch to production mode
- [ ] Monitor and maintain

For support, contact the development team or refer to MonCash documentation at: https://moncashdfs.com/business

**Success!** Your restaurant now has a modern online ordering system with secure payment processing. 🎉